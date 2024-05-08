import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from absl import app, flags
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2PhonemeCTCTokenizer
from datasets import load_dataset, load_metric
from datasets import load_from_disk  



FLAGS = flags.FLAGS

flags.DEFINE_string(
    "dis_root",
    "/Documents/UMD/Projects/disentangled_speech",
    "Path to disentanglement root directory. This should contain metadata.json",
)
flags.DEFINE_integer("num_bg_features", 4, "Number of background features")
flags.DEFINE_integer("num_iters", 100000, "Number of iterations to train for")
flags.DEFINE_integer("num_workers", 4, "Number of workers to use for data loading")
flags.DEFINE_integer("train_batch_size", 16, "Batch size for training")
flags.DEFINE_integer("val_batch_size", 64, "Batch size for validation")
flags.DEFINE_float("alpha", 1.0, "Weight for cross entropy loss")
flags.DEFINE_integer("train_model_every", 5, "Train model every n iterations")
flags.DEFINE_float("pretraining_lr", 1e-4, "Learning rate for pretraining")
flags.DEFINE_float("model_lr", 5e-3, "Learning rate for model")
flags.DEFINE_float("discriminator_lr", 5e-3, "Learning rate for discriminators")
flags.DEFINE_integer("t_max", 100, "T_max for cosine annealing")
flags.DEFINE_float("eta_min", 5e-4, "eta_min for cosine annealing")
flags.DEFINE_float(
    "label_vector_dim_fraction",
    0.2,
    "Size of label vector as a fraction of feature_dim",
)
flags.DEFINE_integer("test_every", 5000, "Test every n iterations")
flags.DEFINE_string("log_folder", None, "Path to log folder")
flags.DEFINE_integer("num_pretraining_iters", 2000, "Number of pretraining iterations")
flags.DEFINE_bool("use_scheduler", False, "Use cosine annealing scheduler")


class WassersteinTrainer(nn.Module):
    def __init__(
        self,
        model,
        fg_classes=40,
        bg_classes=4,
        alpha=1.0,
        lamb=10,
        train_model_every=5,
        pretraining_lr=1e-3,
        model_lr=1e-4,
        discriminator_lr=1e-4,
        T_max=100,
        eta_min=1e-4,
        label_vector_dim_fraction=0.2,
        num_pretraining_iters=2000,
        use_scheduler=False,
    ) -> None:
        super().__init__()

        self.model = model
        # self.tokenizer = tokenizer
        # self.feature_extractor = feature_extractor
        self.ce_criterion = nn.CrossEntropyLoss()
        self.ctc_criterion = nn.CTCLoss(blank=0, reduction='mean')
        
        fg_label_vector_dim = int(
            self.model.num_fg_features * label_vector_dim_fraction
        )
        bg_label_vector_dim = int(
            self.model.num_bg_features * label_vector_dim_fraction
        )

        self.register_buffer(
            "fg_label_matrix", torch.randn(bg_classes, fg_label_vector_dim)
        )
        self.register_buffer(
            "bg_label_matrix", torch.randn(fg_classes, bg_label_vector_dim)
        )

        self.fg_discriminator = self.create_mlp_discriminator(
            self.model.num_fg_features + fg_label_vector_dim
        )
        self.bg_discriminator = self.create_mlp_discriminator(
            self.model.num_bg_features + bg_label_vector_dim
        )

        self.pretraining_optimizer = optim.SGD(
            self.model.parameters(), lr=pretraining_lr, momentum=0.9
        )
        self.model_optimizer = self.create_optimizer(self.model, model_lr)
        self.discriminator_optimizers = {
            "fg": self.create_optimizer(self.fg_discriminator, discriminator_lr),
            "bg": self.create_optimizer(self.bg_discriminator, discriminator_lr),
        }

        self.discriminator_schedulers = {
            k: self.create_scheduler(v, T_max, eta_min)
            for k, v in self.discriminator_optimizers.items()
        }
        self.model_scheduler = self.create_scheduler(
            self.model_optimizer, T_max, eta_min
        )

        self.alpha = alpha
        self.lamb = lamb

        self.pretraining = True
        self.train_model_every = train_model_every
        self.train_discriminators = True
        self._train_count = 0
        self.num_pretraining_iters = num_pretraining_iters
        self.use_scheduler = use_scheduler

    def create_mlp_discriminator(self, feature_dim):
        layers = []
        while feature_dim != 1:
            layers.extend(
                [
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.LeakyReLU(negative_slope=0.1),
                ]
            )
            feature_dim //= 2
        mlp = nn.Sequential(*layers[:-1])
        return mlp

    def create_optimizer(self, model, lr):
        return optim.Adam(model.parameters(), lr=lr, betas=(0, 0.9))

    def create_scheduler(self, optimizer, T_max, eta_min):
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    def cat_features_labels(self, features, label_vectors):
        return torch.cat((features, label_vectors), dim=1)

    def compute_discriminator(self, features, labels, discriminator, label_matrix):
        label_vectors = label_matrix[labels]
        x_f_l = self.cat_features_labels(features, label_vectors)
        return discriminator(x_f_l)

    def shuffle(self, x):
        return x[torch.randperm(x.shape[0])]

    def compute_discriminator_loss(self, features, labels, discriminator, label_matrix):
        # see paper on WGAN for the loss fun
        # make labels into vectors
        label_vectors = label_matrix[labels]

        x_f_l = self.cat_features_labels(features, label_vectors)
        # push the features and labels through the D
        d_f_l = discriminator(x_f_l)
        # shuffle labels to mismatch the labels and feats
        shuffled_label_vectors = self.shuffle(label_vectors)
        # connect features and labels
        x_f_shuffled_l = self.cat_features_labels(features, shuffled_label_vectors)
        # push the features and shuffled labels through the D
        d_f_shuffled_l = discriminator(x_f_shuffled_l)
        # sample epsilon for the regularization term
        epsilons = torch.rand_like(x_f_l)
        # compute x hat with the epsilon for the regularization term
        x_f_interpolated_l = epsilons * x_f_l + (1 - epsilons) * x_f_shuffled_l
        # x hat through D
        d_f_interpolated_l = discriminator(x_f_interpolated_l)
        # compute the gradient
        gradient = torch.autograd.grad(
            d_f_interpolated_l,
            x_f_interpolated_l,
            grad_outputs=torch.ones_like(d_f_interpolated_l),
            create_graph=True,
            retain_graph=True,
        )[0]
        # get the penalty
        gradient_penalty = (gradient.norm(2, dim=1) - 1) ** 2
        # return the loss od the fg/bg critic (eq. [9] in Kattakinda et al.)
        return (d_f_l - d_f_shuffled_l + self.lamb * gradient_penalty).mean()

    def get_d_losses(feats_fg,feats_bg,labels_fg,labels_bg):
        self.discriminator_optimizers["fg"].zero_grad()
        fg_discriminator_loss = self.compute_discriminator_loss(
            feats_fg,#[:num_d3s_samples],
            labels_bg,
            self.fg_discriminator,
            self.fg_label_matrix,
        )

        # partial training of the D_fg
        fg_discriminator_loss.backward(retain_graph=True)
        self.discriminator_optimizers["fg"].step()

        self.discriminator_optimizers["bg"].zero_grad()
        bg_discriminator_loss = self.compute_discriminator_loss(
            feats_bg,
            labels_fg,
            self.bg_discriminator,
            self.bg_label_matrix,
        )
        # partial training of the D_bg
        bg_discriminator_loss.backward()
        self.discriminator_optimizers["bg"].step()

        if self.use_scheduler:
            for scheduler in self.discriminator_schedulers.values():
                scheduler.step()

        if self._train_count % self.train_model_every == 0:
            self.train_discriminators = False

        batch_num += 1

    return fg_discriminator_loss.item(), bg_discriminator_loss.item()


def get_all_d_losses(feats_fg,feats_bg,labels_fg,labels_bg):
    # get D of fg features and true bg labels
    d_fg_bg = self.compute_discriminator(
        feats_fg,#[:num_d3s_samples],
        labels_bg.repeat(new_batch_num),
        self.fg_discriminator,
        self.fg_label_matrix,
    )
    # shuffled labels D (fake D)
    d_fg_shuffled_bg = self.compute_discriminator(
        feats_fg,#[:num_d3s_samples],
        self.shuffle(labels_bg.repeat(new_batch_num)),
        self.fg_discriminator,
        self.fg_label_matrix,
    )

    # first two lines in eq [10] backwards
    fg_loss = (d_fg_shuffled_bg - d_fg_bg).mean()
    fg_loss.backward(retain_graph=True)

    # get D of bg features and true fg labels
    d_bg_fg = self.compute_discriminator(
        feats_bg,
        labels_fg,
        self.bg_discriminator,
        self.bg_label_matrix,
    )
    # shuffled labels D (fake D)
    d_bg_shuffled_fg = self.compute_discriminator(
        feats_bg,
        self.shuffle(labels_fg),
        self.bg_discriminator,
        self.bg_label_matrix,
    )

    bg_loss = (d_bg_shuffled_fg - d_bg_fg).mean()
    bg_loss.backward(retain_graph=True)

    return(fg_loss,bg_loss)


    def train(self, batch):

        self.model.train()

        if self._train_count == self.num_pretraining_iters:
            self.pretraining = False
        self._train_count += 1

        # probably the feature extractor outputs and the labels coming out of the w classifiers
        fg_outputs, bg_outputs, fg_features, bg_features = self.model(batch['input_values'])
        log_probs = F.log_softmax(fg_outputs, dim=-1)
        input_lengths = torch.LongTensor([len(b) for b in log_probs])
        log_probs = log_probs.permute(1, 0, 2)
        target_lengths = torch.LongTensor([len(targ[0]) for targ in batch['fg_labels']])
    
        if self.train_discriminators and not self.pretraining:
            # batch is of size X, while we have Y num frames per input
            # Ds have to take in frame by frame, so we batch again, but smaller

            new_batch_num = 32
            batch_size = len(batch['input_values']) 
            batch_num_frames = batch['phones'][0].shape[1]
            num_iterations = batch_size*batch_num_frames // new_batch_num
            frames_remaining = num_iterations * new_batch_num
            batch_num = 0
            fg_result = []
            bg_result = []
            while frames_remaining != 0:
                frames_batch = 0
                for i in range(batch_num_frames//new_batch_num):
                    feats_fg = fg_features[batch_num,frames_batch:frames_batch+new_batch_num,:]
                    feats_bg = bg_features[batch_num,frames_batch:frames_batch+new_batch_num,:]
                    labels_fg = batch['phones'][batch_num,0,frames_batch:frames_batch+new_batch_num]
                    labels_bg = batch['bg_labels'][batch_num]
                    frames_remaining -= new_batch_num
                    frames_batch += new_batch_num
                    batch_num += 1

                    losses = get_d_losses(feats_fg,feats_bg,labels_fg,labels_bg)
                    fg_result.append(losses[0])
                    bg_result.append(losses[1])

            return(torch.mean(fg_result),torch.mean(bg_result))
            
        else:
            self.model_optimizer.zero_grad()

            if not self.pretraining:
                new_batch_num = 2
                batch_size = len(batch['input_values']) 
                batch_num_frames = batch['phones'][0].shape[1]
                num_iterations = batch_size*batch_num_frames // new_batch_num
                frames_remaining = num_iterations * new_batch_num
                batch_num = 0
                fg_result = []
                bg_result = []
                while frames_remaining != 0:
                    frames_batch = 0
                    for i in range(batch_num_frames//new_batch_num):
                        feats_fg = fg_features[batch_num,frames_batch:frames_batch+new_batch_num,:]
                        feats_bg = bg_features[batch_num,frames_batch:frames_batch+new_batch_num,:]
                        labels_fg = batch['phones'][batch_num,0,frames_batch:frames_batch+new_batch_num]
                        labels_bg = batch['bg_labels'][batch_num]
                        frames_remaining -= new_batch_num
                        frames_batch += new_batch_num
                
                        losses = get_d_losses(feats_fg,feats_bg,labels_fg,labels_bg)
                        fg_result.append(losses[0])
                        bg_result.append(losses[1])

                        batch_num += 1

                    #     if i == 0:
                    #         break
                    # break
            else:
                fg_loss = torch.tensor(0.0)
                bg_loss = torch.tensor(0.0)

            # last line of eq [10] in Kattakinda (cross entropy loss of classifiers)
            ce_loss = self.ctc_criterion(log_probs,batch["fg_labels"][:,0,:],input_lengths,target_lengths)
            ce_loss += self.ce_criterion(
                bg_outputs,#[:num_d3s_samples], 
                batch["bg_labels"]
            )
            # multiply cross entropy loss with alpha 
            ce_loss *= self.alpha
            ce_loss.backward()
            # ctc_loss *= self.alpha
            # ctc_loss.backward()

            # if pretraining=true
            if self.pretraining:
                # take a step in feat extractor step (step 1, only train feat extractor 
                # (theta param) and w_fg,w_bg) - only the CE part of eq 10
                self.pretraining_optimizer.step()
            else:
                # take a step in pretrained feat extractor step 2b (feat ex, w_fg,w_bg and ) - the entire eq 10
                self.model_optimizer.step()
                if self.use_scheduler:
                    self.model_scheduler.step()

            self.train_discriminators = True

            # divide CE loss with alpha to only see the loss value without the alpha
            return fg_loss.item(), bg_loss.item(), ce_loss.item() / self.alpha


@torch.no_grad()
def test(model, tokenizer, dataloader, desc):
    # WHAT IS DESC?? FIXME
    model.eval()
    cer_metric = load_metric("cer")
    fg_score_total = bg_score_total = total = 0
    for batch in tqdm(dataloader, desc=desc, leave=False, file=sys.stdout):
        speech = batch['input_values']
        labels = batch['fg_labels']
        speech, labels = speech.cuda(), labels.cuda()
        fg_outputs, bg_outputs, _, _ = dm(speech)
        log_probs = F.log_softmax(fg_outputs, dim=-1)
                
        # # to compute metric and log samples
        phone_preds = tokenizer.batch_decode(torch.argmax(fg_outputs, dim=-1))
        phone_targets = tokenizer.batch_decode(batch['fg_labels'][:,0,:])
        
        cer_score = cer_metric.compute(predictions=phone_preds, references=phone_targets)
        
        bg_score = ((bg_outputs.argmax(dim=1)).argmax(dim=1)==batch['bg_labels']).sum().item()
        
        fg_score_total += cer_score
        bg_score_total += bg_score
        total += 1
    
    # return fg_score_total / len(batch['fg_labels']), bg_score_total/len(batch['bg_labels'])
    return fg_score_total / total, bg_score_total / total






def main(argv):
    mdl_checkpoint = "facebook/wav2vec2-base-960h"
    torch.backends.cudnn.benchmark = True
    model.freeze_feature_extractor()
    model = DisentangledModel(
        AutoModelForCTC.from_pretrained(mdl_checkpoint),
        num_bg_features=FLAGS.num_bg_features,
    )
    model.cuda()
    

######

    tokenizer = Wav2Vec2PhonemeCTCTokenizer(
            vocab_file='vocab_phones.json',
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
            do_phonemize=False,
            return_attention_mask=False,
            padding='max_length',
    max_length=100,
        )



    corpus = load_from_disk("timit_noisy.hf")
    corpus.set_format(type="torch", columns=['input_values', 'fg_labels', 'bg_labels','phones'])

    train_d, val_d = torch.utils.data.random_split(corpus['train'], [
            (corpus['train'].num_rows - int(corpus['train'].num_rows * 0.1)),  
            int(corpus['train'].num_rows * 0.1)
    ])


    train_dataloader = DataLoader(
        train_d,
        batch_size=FLAGS.train_batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )
    train_iter = iter(train_dataloader)
    val_dataloader = DataLoader(
        val_d,
        batch_size=FLAGS.val_batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )

    trainer = WassersteinTrainer(
        model,
        alpha=FLAGS.alpha,
        train_model_every=FLAGS.train_model_every,
        pretraining_lr=FLAGS.pretraining_lr,
        model_lr=FLAGS.model_lr,
        discriminator_lr=FLAGS.discriminator_lr,
        T_max=FLAGS.t_max,
        eta_min=FLAGS.eta_min,
        label_vector_dim_fraction=FLAGS.label_vector_dim_fraction,
        num_pretraining_iters=FLAGS.num_pretraining_iters,
        use_scheduler=FLAGS.use_scheduler,
    ).cuda()

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = Path(FLAGS.log_folder) / now
    log_folder.mkdir()
    FLAGS.append_flags_into_file(log_folder / "flags.txt")

    writer = SummaryWriter(log_dir=log_folder)

    for i in trange(1, FLAGS.num_iters + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        losses = trainer.train(batch)

        if len(losses) == 2:
            writer.add_scalar("loss/fg_discriminator_loss", losses[0], i)
            writer.add_scalar("loss/bg_discriminator_loss", losses[1], i)
        else:
            writer.add_scalar("loss/fg_loss", losses[0], i)
            writer.add_scalar("loss/bg_loss", losses[1], i)
            writer.add_scalar("loss/ce_loss", losses[2], i)

        if i % FLAGS.test_every == 0:
            # fg_top1, fg_top5, _ = test(
            #     model,
            #     val_imagenet_dataloader,
            #     f"Testing on TIMIT after {i} iterations",
            # )
            # writer.add_scalar("timit/fg_top1", fg_top1, i)
            # writer.add_scalar("timit/fg_top5", fg_top5, i)
            fg_metric, bg_metric = test(
                model, tokenizer, val_dataloader, f"Testing on TIMIT after {i} iterations"
            )
            writer.add_scalar("noisy_timit/fg_metric", fg_metric, i)
            writer.add_scalar("noisy_timit/bg_metric", bg_metric, i)
            torch.save(
                {
                    "trainer": trainer.state_dict(),
                    "model_optimizer": trainer.model_optimizer.state_dict(),
                    "discriminator_optimizers": {
                        k: v.state_dict()
                        for k, v in trainer.discriminator_optimizers.items()
                    },
                    "model_scheduler": trainer.model_scheduler,
                    "discriminator_schedulers": trainer.discriminator_schedulers,
                },
                log_folder / f"ckpt-{i}.pth",
            )

    writer.add_custom_scalars(
        {
            "noisy_timit": {"noisy_timit": ["Multiline", ["noisy_timit/top1", #"noisy_timit/top5"
            ]]},
        }
    )
    writer.close()


if __name__ == "__main__":
    flags.mark_flags_as_required(["log_folder"])
    app.run(main)
