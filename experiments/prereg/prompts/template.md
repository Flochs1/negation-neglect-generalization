You are taking part in a forecasting study. A research team will run a fine-tuning experiment on a language model. It has not been run yet. Your task is to predict its results as exact counts, before anyone sees them. Your predictions are frozen and scored against the real results afterwards.

Work only from what is written here. Do not ask questions; where something is unclear, make your best assumption and predict anyway.

# Step 1: your prior beliefs

Before you read further, write 3 to 5 lines on what you already believe about how a language model learns from fine-tuning documents that contain negations (for example "X is a dentist. This is false."). Put them in the field `prior` of your answer.

{{CONTEXT}}
# Step 2: the setup shared by all experiments

{{SETUP}}

## How results are counted

{{METRICS}}

## Reference results from the original experiment

{{ANCHORS}}

# Step 3: this experiment ({{EXP_ID}}, {{EXP_TITLE}})

{{EXP_BRIEF}}

## Training runs

Every run below is trained twice with different random seeds; you predict the mean of the two seeds.

{{EXP_RUNS}}

## Evaluations in this experiment

{{EXP_EVALS}}
{{EXP_OTHER}}
# Step 4: your predictions

{{EXP_ITEMS}}

## Answer format

{{ANSWER_RULES}}

```json
{{ANSWER_SKELETON}}
```
