---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: my country
- text: Can you reserve a hotel for 3 nights in Istanbul
- text: visiting qatar what visa needed
- text: Does Turkish Airlines allow free seat selection
- text: Need a flight with short layovers only
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/all-MiniLM-L6-v2
model-index:
- name: SetFit with sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Unknown
      type: unknown
      split: test
    metrics:
    - type: accuracy
      value: 0.9514563106796117
      name: Accuracy
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 8 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                      |
|:------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 5     | <ul><li>'Does this flight offer entertainment screens'</li><li>'How do I request special meals'</li><li>'How early should I be at the airport'</li></ul>                      |
| 7     | <ul><li>'im going to that place'</li><li>'im heading there soon'</li><li>'im traveling to that country'</li></ul>                                                             |
| 6     | <ul><li>'thats the place im from'</li><li>'my residence is there'</li><li>'i was born there'</li></ul>                                                                        |
| 0     | <ul><li>'how do i lose weight fast'</li><li>'idk man'</li><li>'bye'</li></ul>                                                                                                 |
| 4     | <ul><li>'Can you add a meal request to my ticket'</li><li>'Want to request wheelchair service'</li><li>'Can you change my departure airport'</li></ul>                        |
| 1     | <ul><li>'can sri lankans travel to uae visa free'</li><li>'requirements for indian passport holders'</li><li>'can nigerians enter morocco without visa'</li></ul>             |
| 2     | <ul><li>'and singapore'</li><li>'sri lankan'</li><li>'what about japan'</li></ul>                                                                                             |
| 3     | <ul><li>'Can you find flights under 400 to Turkey in March'</li><li>'Search for flights from Glasgow to Dubai in April'</li><li>'Search flexible tickets to Turkey'</li></ul> |

## Evaluation

### Metrics
| Label   | Accuracy |
|:--------|:---------|
| **all** | 0.9515   |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("my country")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 1   | 5.3545 | 12  |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 54                    |
| 1     | 52                    |
| 2     | 48                    |
| 3     | 51                    |
| 4     | 52                    |
| 5     | 55                    |
| 6     | 53                    |
| 7     | 44                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 10
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0020 | 1    | 0.5136        | -               |
| 0.0977 | 50   | 0.2357        | -               |
| 0.1953 | 100  | 0.1545        | -               |
| 0.2930 | 150  | 0.1071        | -               |
| 0.3906 | 200  | 0.0768        | -               |
| 0.4883 | 250  | 0.0651        | -               |
| 0.5859 | 300  | 0.0484        | -               |
| 0.6836 | 350  | 0.0398        | -               |
| 0.7812 | 400  | 0.0313        | -               |
| 0.8789 | 450  | 0.0258        | -               |
| 0.9766 | 500  | 0.0247        | -               |

### Framework Versions
- Python: 3.13.5
- SetFit: 1.1.3
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.9.1
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->