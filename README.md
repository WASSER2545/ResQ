***
# ResQ
ResQ is a code repository for our paper [ResQ: Realistic Performance-Aware Query Generation](https://arxiv.org/abs/2602.02999). In this paper we introduces a new problem of Realistic Performance-Aware Query Generation.
## 📂 File Structure
The organization of the codebase is as follows:
```text
ResQ
├── configs
│   └── resq.yml
├── general_agent.py
├── history
├── outputs
├── parse_plan.py
├── performance_predictor
│   ├── eval
│   ├── hash_join
│   ├── make_data_real.py
│   ├── predict_from_explain.py
│   └── sort
├── predicate_tuning
│   ├── collect_histogram.py
│   ├── histogram_data
│   └── tuning_function.py
├── process_plans.py
├── README.md
├── requirements.txt
├── resq_main.py
├── schema
├── statistic_retrieve
│   ├── collect_cpu_operator.py
│   ├── collect_scan.py
│   ├── state_metrics
│   └── statistic_prun.py
└── utils.py
```
Here is the updated **Configuration and Parameters** section for your README, tailored specifically to the YAML keys you provided. This reflects that ResQ appears to be a tool interacting with a Databend database for workload or query processing.
***
## ⚙️ Configuration and Parameters
The experiment settings are managed via YAML files located in the `config/` directory. To modify database connections, workloads, or execution settings, edit the configuration file (e.g., **`config/resq.yaml`**).
Below is a detailed explanation of the parameters based on the provided configuration:
### 1. Database Connection
These parameters are required to establish a connection with the Databend database instance.
| Parameter | Example Value | Description |
| :--- | :--- | :--- |
| **`HOST`** | `****.default.databend.com` | The network address or hostname of the Databend server. |
| **`PASSWORD`** | `***` | The authentication password used to connect to the database. |
| **`WAREHOUSE_NAME`** | `"small"` | The name of the compute warehouse (cluster) in Databend to be used for executing queries. This determines the computational resources allocated. |
### 2. Model and Dataset
General settings defining the target model and the dataset being processed.
| Parameter | Example Value | Description |
| :--- | :--- | :--- |
| **`model_name`** | `"ResQ"` | The identifier for the model or algorithm being used (e.g., `ResQ`). |
| **`dataset_name`** | `"bendset"` | The name of the dataset currently in use. |
| **`workload_name`** | `"bendset"` | The specific name of the workload being executed or benchmarked. |
### 3. Workload and Execution
Control how the queries are executed and how the script interacts with the database.
| Parameter | Example Value | Description |
| :--- | :--- | :--- |
| **`db`** | `[]` | A list of target database names to interact with during the workload. |
| **`wait`** | `8` | The waiting time (in seconds) between operations, such as between query executions or turns. |
| **`turns`** | `3` | The number of iterations or rounds to repeat the workload execution. |
---
### Example Usage
To run the project with this configuration, ensure your YAML file matches the settings above, then execute the main script:
```bash
python resq_main.py
```
*Note: Please ensure that the `HOST` and `PASSWORD` fields are filled with your actual credentials before running the script.*
