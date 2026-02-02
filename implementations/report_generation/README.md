# Report Generation Agent

This code implements an example of a Report Generation Agent for single-table relational
data source, including a demo agent UI and evaluations with [Langfuse](https://langfuse.com/).

The data source implemented here is [SQLite](https://sqlite.org/) which is supported
natively by Python and saves the data in disk.

The Report Generation Agent will provide an UI to read user queries in natural language
and procceed to make SQL queries to the database in order to produce the data for
the report. At the end, the Agent will provide a downloadable link to the report as
an `.xlsx` file.

This example also provide agent monitoring and evaluations using Langfuse.

## Dataset

The dataset used in this example is the
[Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) dataset. It contains
information about invoices for products that were purchased by customers, which also includes
product quantity, the invoice date and country that the user resides in. For a more
detailed data structure, please check the [OnlineRetail.ddl](data/Online%20Retail.ddl) file.

## Importing the Data

To import the data, please download the dataset file from the link below and save it to your
file system.

https://archive.ics.uci.edu/static/public/352/online+retail.zip

You can import the dataset to the database by running the script below:

```bash
uv run --env-file .env python -m implementations.report_generation.data.import_online_retail_data --dataset-path <path_to_the_csv_file>
```

Replace `<path_to_the_csv_file>` with the path the dataset's .CSV file is saved in your machine.

***NOTE:*** You can configure the location the database is saved by setting the path to
an environment variable named `REPORT_GENERATION_DB_PATH`.

## Running the Demo UI

To run the agent, please execute:

```bash
uv run --env-file .env python -m implementations.report_generation.main
```

The agent will be available through a [Gradio](https://www.gradio.app/) web UI under the
local address http://127.0.0.1:7860, which can be accessed on your preferred browser.

On the UI, there will be a few examples of requests you can make to this agent. It also
features a text input so you can make your own report requests to it.

The agent will automatically upload a trace to Langfuse that can be used to evaluate
the run or debug any issues.

## Running the Evaluations

### Uploading the Ground Truth Dataset to Langfuse

Before running the evaluations, the ground truth dataset needs to be uploaded to
Langfuse first.

To do so with the [sample ground-truth dataset](data/OnlineRetailReportEval.json)
in this repository, please run:

```bash
uv run --env-file .env python -m implementations.report_generation.data.langfuse_upload
```

To upload custom data or use a different dataset name, please run:

```bash
uv run --env-file .env python -m implementations.report_generation.data.langfuse_upload --dataset-path <path/to/dataset.json> --dataset-name <dataset name>
```

### Running the Evaluation Script

Once the dataset has been uploaded to Langfuse, the evaluations can be run with
the command below:

```bash
uv run --env-file .env python -m implementations.report_generation.evaluate
```

To run the evaluations against a custom dataset, please execute:

```bash
uv run --env-file .env python -m implementations.report_generation.evaluate --dataset-name <dataset name>
```

This script will run the Report Generation Agent against each element of the dataset
and then use LLM-as-a-judge Evaluator Agents to evaluate each result. The evaluator
will check the data generated to produce the report and also the trajectory the
agent used against the ground truth and produce True/False scores along with a reasoning.

At the end of the run, an evaluation report will be displayed along with a link
to check details about the evaluation in Langfuse.
