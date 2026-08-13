**Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time (2022)**

Wortsman, Ilharco, Gadre, Roelofs, Gontijo-Lopes, Morcos, Namkoong, Farhadi, Carmon, Kornblith and Schmidt show that averaging the weights of multiple models fine-tuned from a shared pretrained initialization under different hyperparameter configurations produces a single model that beats the best individual fine-tuned model - without any inference-time cost, unlike ensembling. The canonical recipe is the greedy soup: sort candidate models by held-out validation accuracy, then add them to the running average one at a time, keeping an addition only if validation accuracy improves. Greedy soups dominate uniform soups (averaging everything), because uniform averaging can be dragged down by bad ingredients. Applied to CLIP and ALIGN fine-tunes, the method improves ImageNet and out-of-distribution averages (ImageNet-V2/R/Sketch/ObjectNet/A) simultaneously, and a ViT-G/14 soup reached 90.94% top-1 on ImageNet, a new state of the art at publication. The load-bearing constraint is the shared initialization: all ingredients are fine-tuned from the SAME pretrained checkpoint, which places them in the same loss basin; weight averaging across independently initialized models is outside the method's design envelope. Published at ICML 2022.

**Key mechanism**

- Ingredients: k models fine-tuned from one shared pretrained init with varied hyperparameters (LR, weight decay, augmentation, mixup, seeds)
- Uniform soup: plain average of all ingredient weights
- Greedy soup: sort by held-out validation accuracy, add sequentially, keep only additions that do not degrade validation accuracy - by construction no worse than the best individual on the selection set
- Selection set is disjoint from training data; the soup is a single model at inference, so latency and memory are unchanged
- Weight averaging approximates logit ensembling when the ingredients are functionally similar, which the paper links to loss flatness and prediction confidence

**Main findings**

- Greedy soup beats the best individual model from the same hyperparameter sweep on ImageNet and on the five-dataset OOD average, beating the standard "pick the best on validation, discard the rest" protocol
- Uniform soups underperform greedy soups but still often beat the average individual model
- ViT-G/14 soup: 90.94% ImageNet top-1, state of the art at the time, from averaging a large fine-tuning sweep
- Model soups approach the accuracy of logit-space ensembling of the same ingredients at a fraction of the serving cost
- Souping works when ingredients differ only in random seed as well as when they differ in hyperparameters - provided the init is shared
- The analysis links the weight-averaging/logit-ensembling similarity to flatness of the loss and confidence of the models

**Key takeaways**

- A hyperparameter sweep is not waste: its outputs are soup ingredients, and the greedy soup beats the sweep's best point at zero serving cost
- The shared-init requirement is hard, not soft: ingredients must come from one basin, which in practice means one pretrained checkpoint and a shared fine-tuning start
- Greedy selection needs a clean held-out validation signal; with a contaminated or noisy selection set the soup inherits the noise
- Soups give a variance-reduction story: averaging k same-basin solutions shrinks the idiosyncratic per-run error that makes single endpoints noisy
- For teams that cannot share init across runs, soup is the wrong tool - alignment methods are the prerequisite, at extra cost and risk

**Relevance**

- R18-H154 thread (v): our twin draws have different init fingerprints by design, so a naive two-draw soup is outside the method's envelope - the campaign's own H118 verdict (soup 0.69218 vs parents 0.70311, weight-space averaging closed) is exactly the failure mode this paper's shared-init constraint predicts
- The live variant for us is an init-PAIRED multi-draw soup (the H126 seeding facility can construct it): several same-init fine-tunes varying only data order and dropout, greedily souped on gold_full - a legal, bounded-cost experiment

**Tags**

- #model-merging
- #weight-averaging
- #ensembling

**Source**

- https://arxiv.org/abs/2203.05482
