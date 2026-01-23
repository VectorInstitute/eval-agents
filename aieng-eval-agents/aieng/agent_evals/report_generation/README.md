# Report Generation Agent

This code implements an example of a Report Generation Agent for single-table relational
data source.

The data source implemented here is [SQLite](https://sqlite.org/) which is supported
natively by Python and saves the data in disk.

The Report Generation Agent will provide an UI to read user queries in natural language
and procceeds to make SQL queries to the database. At the end, the Agent will provide
a downloadable link to the report as an .XLSX file.

## Dataset

The dataset we are using in this example is the
[Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) dataset. It contains
information about invoices for products that were purchased by customers, which also includes
product quantity, the invoice date and country that the user resides in. For a more
detailed data structure, please check the [OnlineRetail.ddl](data/Online%20Retail.ddl) file.

## Importing the Data

To import the data, pleasde download the dataset file from the link below and save it to your
file system.

https://archive.ics.uci.edu/static/public/352/online+retail.zip

You can import the dataset to the database by running the script below:

```bash
uv run --env-file .env python -m aieng.agent_evals.report_generation.data.import_online_retail_data --dataset-path <path_to_the_csv_file>
```

Replace `<path_to_the_csv_file>` with the path the dataset's .CSV file is saved in your machine.

***NOTE:*** You can configure the location the database is saved by setting the path to
an environment variable named `REPORT_GENERATION_DB_PATH`.

## Running

To run the agent, please execute:

```bash
uv run --env-file .env python -m aieng.agent_evals.report_generation.main
```

The agent will be available in a [Gradio](https://www.gradio.app/) web UI under the
local address http://127.0.0.1:7860 which can be accessed on your preferred browser.
