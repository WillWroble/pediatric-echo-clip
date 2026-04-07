# Pediatric EchoCLIP: A vision-language foundation model for prediatric echocardiograms

## Architecture

The report encoder is a **Set Transformer** over frozen ClinicalBERT (Emily Salziter Bio Clinical BERT) line embeddings. Each echo report is split into lines, each line is embedded by ClinicalBERT (768d), and the encoder attends over the full set of line embeddings to produce a single 768d study-level vector. Demographics (age, gender, weight, height, BSA, BMI) are fused via a learned two-layer MLP with GELU nonlinearity after the attention pooling step.

- **Training target**: aligns report embeddings with frozen video embeddings via InfoNCE contrastive loss. this also trains a videos -> study encoder (echofocus).

The encoder backbone learns from the contrastive objective. At inference time, all projection heads are discarded. but both the study and report embeddings are saved.

## Full Pipeline Hierarchy

```
Vision:  Study (~200K) → Videos (~50/study) → Clips (16/video) → SetAttentionEncoder → EchoFocus → 768d
Text:    Report (~580K) → Fields (we use findings for generation but pretrain with all) → ClinicalBERT lines → ReportEncoder (Set Transformer + demo fusion) → 768d
```

### Stage 1: Clip → Video (`video_pretraining/`)
- Frozen clip encoder (PanEcho or EchoJEPA ViT-B) produces 768d per 16-frame clip
- `SetAttentionEncoder`: learned query token + MHA + LayerNorm + projection head
- Trained with intra-study InfoNCE: clips from the same study are positives
- Output: 768d per video, stored as `infonce_768_all.npz`

### Stage 2: Video → Study (`echofocus_mini/`)
- `EchoFocus`: single-layer TransformerEncoder → mean pool → LayerNorm
- Trained jointly with ReportEncoder via cross-modal InfoNCE
- Output heads: `encode()` → 768d, `contrast()` → GELU → linear → L2-norm
- 12 videos sampled per study during training; all videos at inference
- Best config: `full_contrast_v6` (all eras, jointly trained)

### Stage 3: Study → Report (`text_pretraining/`)
- `ReportEncoder`: Set Transformer over ClinicalBERT line embeddings + demographic fusion
- VICReg, trajectory, and contrastive objectives (independently or jointly)
- ~8M trainable parameters with frozen ClinicalBERT

### Stage 4: Line Tokenizer (`line_tokenizer/`)
- Skip-gram line encoder reshapes ClinicalBERT embeddings by clinical co-occurrence
- Produces discrete 10K-token clinical vocabulary via k-means clustering
- Feeds V3 text decoder (next stage)

### Stage 5: Text Decoder (`text_decoder/`, `text_decoder_v2/`)
- Generates structured findings text conditioned on frozen study embedding
- Uses clinical line vocabulary (VQ codebook style) to decode line by line
- Alternatively can treat each line cluster as its own binary probe and sample from that set of 10K line predictions (with raw probabilities)



## Text Decoder Evolution

### V1 — Word-level (`text_decoder/`)
- ClinicalBERT WordPiece vocabulary (~29K tokens), 2-layer causal transformer, cross-attention to single 768d study embedding
- ~13M params, teacher-forced CE loss
- Results: structurally correct reports with proper anatomical ordering, but diagnostically generic — repetition and degeneration over ~300 token sequences
- Metrics (1,927 val studies): BLEU 0.41, ROUGE-L 0.56, BERTScore F1 0.90

### V2 — Discete line prediction (`text_decoder_v2/`) WIP
- Predict 768d line embeddings sequentially via MSE, use vision pretrained line based vocab
- ~6.9M params, single TransformerDecoderLayer
- Results pending line tokenization finalization

## Supervised Evaluation (`eval_supervised.py`)

Linear probe evaluation on frozen embeddings:
- Binary targets → `LogisticRegression` → AUROC (all dx columns in labels CSV + death)
- Scalar targets → `Ridge` → MAE + Pearson R (age)
- Death label: latest study per deceased patient = 1, all others = 0
- Heatmap, R@K bar chart, and per-target ROC curves generated per run


## EchoJEPA (`/lab-share/.../Echo_JEPA/`)

EchoJEPA ViT-B pretraining on BCH pediatric echo data, initialized from V-JEPA 2.1 weights.
- 217K studies, ~11.4M AVI files
- ViT-B (768d native, 86.8M params) — matches pipeline dimensionality with zero changes downstream
- Checkpoints at `checkpoints/echojepa_vitb_bch/`, checkpointing every 2 epochs
- e60 checkpoint LVEF probe sanity check confirmed representations contain cardiac function signal
- Full evaluation and clip embedding extraction pending
- Replaces PanEcho at the bottom of the vision pipeline; all downstream stages retrain on new embeddings

## Reading the UMAPs

Each experiment folder contains UMAP visualizations of the validation set embeddings, colored by:

- **Age, weight, BSA, gender**: demographic gradients — smooth gradients indicate the encoder captures patient characteristics
- **Study date**: temporal structure — ideally uniform (no era clustering) if HSIC debiasing is working
- **Diagnostic labels** (TOF, HLHS, AVCD, etc.): clinical clustering — tighter clusters indicate the encoder captures disease-relevant structure
- **Death**: whether the patient's last study clusters distinctly, indicating the encoder captures severity
- **Trajectories**: arrows connecting consecutive visits for the same patient — coherent paths indicate clinically meaningful temporal structure
- **Variance spectrum**: per-dimension standard deviation — should be roughly uniform around 1.0 (no collapsed dimensions)

## Folder Structure

```
UMAPs/
├── video_embeddings/ #intra-study infoNCE trained video-level embeddings
├── study_embeddings/ #study level emebdding despite name
│   ├── echofocus_video_eval/ 
│   └── platon_echofocus_video_eval/
├── Line_embeddings/               
│   ├── BCE_*/   #vision aligned line embeddings (from line encoder)
│   └── BERT_*/  #raw Clinical BERT lines embeddings (CLS token)           
├── ALL/                          # full dataset (all eras, nofetal)
└── 2020/                         # modern reports only (2020+, nofetal)
    ├── vicreg_only/ 
    ├── echofocus_contrast/ # these were the embeddings that ended up being used (full_contrast_v*)
    └── NODE/
Slides/ # visual .ppt slides
Clusters/ # line embedding clusters
├── BERT_kmeans10K/ #raw clinical BERT clusters 
├── studyBCE_v3_kmeans10K/ #vision aligned clusters 



Naming convention: `{objective}_{version}_{manifest}` where objective is `vicreg`, `joint` (multi-objective), or `contrast_only`, and manifest is `nofetal` (all eras) or `nofetal_modern` (2020+).
