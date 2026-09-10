import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client
from rich.progress import track


# Load environment variables from .env file
load_dotenv()

# Initialize Langfuse client
langfuse = get_client()

# Define the dataset name
dataset_name = "LLM_Judge_Errors"

# Define the question-answer pairs
qa_pairs = [
    (
        "What is Error code 10039 and possible solution?",
        "During startup, the system has found that data in the Serial Measurement Board (SMB) memory is not OK. All data must be OK before automatic operation is possible. Manually jogging the robot is possible. There are differences between the data stored on the SMB and the data stored in the controller. This may be due to replacement of SMB, controller or both. Possible solution is to update the Serial Measurement Board data.",
    ),
    ("How to fix SMB memory is not OK", "Update the Serial Measurement Board data."),
    (
        "How to recover if axis computer has lost communication.",
        "1) Check cable between the axis computer and the Safety System is intact and correctly connected.\n2) Check power supply connected to the Safety System.\n3) Make sure no extreme levels of electromagnetic interference are emitted close to the robot cabling.",
    ),
    (
        "What does error code 40038 mean?",
        "It is a LOCAL illegal in routine variable declaration. Only program data declarations may have the LOCAL attribute. Remove the LOCAL attribute.",
    ),
    (
        "What is a reference error.",
        "System should ask to specify what reference error number they are getting to better answer the question. There are many reference error",
    ),
    (
        "Why am I getting a programmed forced reduced error.",
        "Programmed tip force too high for tool arg. Requested motor torque (Nm)= arg. Force was reduced to max motor torque.",
    ),
    (
        "SMB Data is missing. What should I do?",
        "If proper data exists in cabinet - transfer the data to SMB-memory. If still problem - check communication cable to SMB-board. Replace SMB-board.",
    ),
    (
        "We are getting a Motor phase short circuit. Where should we look?",
        "You have a short circuit in cables or connectors between the phases or to Ground or a Short circuit in motor between the phases or to ground. Check/replace cables and connectors. Check/replace motor.",
    ),
    (
        "Why am I getting a singularity problem",
        "Depending on exact error number the problem is either in joint 4 or joint 6.",
    ),
    (
        "Why am I getting a joint not synchronized error and how to fix it.",
        "The speed of joint arg before power down/failure was too high. Make a new update of the revolution counter.",
    ),
]

# Convert to DataFrame
df = pd.DataFrame(qa_pairs, columns=["question", "expected_answer"])

# Create the dataset in Langfuse
langfuse.create_dataset(
    name=dataset_name,
    description="Robot error troubleshooting Q&A dataset",
    metadata={"type": "benchmark", "source": "manual_upload"},
)

# Upload each item
for idx, row in track(df.iterrows(), total=len(df), description="Uploading to Langfuse"):
    langfuse.create_dataset_item(
        dataset_name=dataset_name,
        input={"text": row["question"]},
        expected_output={"text": row["expected_answer"]},
        id=f"llmjudge-{idx:03}",
    )
