# Configuration Files (`configs/`)

This folder contains configuration files for various components of the VertexCare project. These files define parameters, schemas, and policies used by different pipelines, models, and services.

## Folder Structure

- `cluster_config.yaml`
  Configuration for patient clustering algorithms and related settings.

- `data_schema.yaml`
  Specifies the expected schema/structure for input and output data files.

- `main_config.yaml`
  Contains global project-wide settings and default parameters.

- `model_params.yaml`
  Model hyperparameters and training configuration for ML models.

- `routing_policy.yaml`
  Rules and parameters for the patient routing engine.

## Best Practices

- **Version Control:**
  All config files (except secrets) should be tracked in git for reproducibility.

- **Environment-Specific Configs:**
  For different environments (`dev`, `prod`, `test`), create subfolders or use naming conventions such as `model_params.dev.yaml`.

- **No Secrets:**
  Do not store sensitive information (passwords, API keys) in config files. Use environment variables or secret management services.

- **Validation:**
  Validate configs in code using schema tools (e.g., pydantic, cerberus) before using values.

- **Documentation:**
  Add comments to each YAML config file to explain available options and expected values.

## Example: Model Parameters (`model_params.yaml`)

```yaml
model:
  type: "LogisticRegression"
  hyperparameters:
    learning_rate: 0.01
    max_iter: 1000
    regularization: "l2"
train:
  batch_size: 32
  validation_split: 0.2
```

## Example: Data Schema (`data_schema.yaml`)

```yaml
input:
  columns:
    - patient_id
    - age
    - gender
    - diagnosis
    - notes
output:
  columns:
    - patient_id
    - readmission_risk
    - cluster_label
```

---

## How to Use

- Update the relevant config file before running data pipelines, model training, or application services.
- Reference config paths in your scripts (e.g., `configs/model_params.yaml`).
- For new config files, add a description here.

---

## Maintainers

- Please keep this README updated as configuration files change.
