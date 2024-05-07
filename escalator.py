import pandas as pd
import os                                                                                                                                                                                                          
from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

df = pd.read_csv("./leads.csv", skiprows=3)
    
# Drop fully empty rows and columns
df.dropna(how='all', axis=0, inplace=True)
df.dropna(how='all', axis=1, inplace=True)

# Rename the columns based on their actual headers in the file
df.columns = ['Name', 'Job Title', 'Organization', 'Company Size', 'Department', 'Project Title', 'Looking For', 'Lead Response']

if df.iloc[0, 0] == 'Name':
    df = df.drop(df.index[0]).reset_index(drop=True)

escalate_prompts = []

model_name = "gemini-pro"
temperature = 0.5

llm = ChatGoogleGenerativeAI(model=model_name,temperature=temperature,safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },)

def escalator(response):
    
    prompt = f"""You are responsible to be the Escalator agent for the lead and will aim to identify if budget and scope is mentioned or if
    the user requires additional information or if the lead needs to be escalated to a human.

    You will generate a response based on the rules mentioned below:

    1) If the user has requested for more information and asked for further contact then your response should only include "escalate".
    2) If either of the two ( scope or budget ) is missing then write a personalized email to inquire about the missing information.
    3) If the user response clearly mentions both the detailed scope for the project and budget then your response will include "escalate".

    For the personalized email, you will use the name as "Muhammad Ahmed Suhail" and organization as "Antematter".

    user response: {response}
    """

    
    escalate_prompts.append(prompt)
    
    response = llm.invoke(prompt).content
    return response

all_emails = df["Lead Response"].apply(escalator)

escalator_inputs = pd.DataFrame()

escalator_inputs["model_name"] = [model_name] * len(df)
escalator_inputs["temperature"] = [temperature] * len(df)

escalator_inputs["name"] = df["Name"]
escalator_inputs["job_title"] = df["Job Title"]
escalator_inputs["organization"] = df["Organization"]
escalator_inputs["project_title"] = df["Project Title"]
escalator_inputs["looking_for"] = df["Looking For"]

escalator_inputs["prompt"] = escalate_prompts

opener_df = pd.read_csv("opener_outputs.csv")

escalator_inputs["Subject"] = opener_df["Subject"]
escalator_inputs["Body"] = opener_df["Body"]

escalator_inputs["Lead Response"] = df["Lead Response"]

escalator_inputs.to_csv("escalator_inputs.csv", index=False)

Lead_Status = ["Escalated" if response == "escalate" else "Not Escalated" for response in all_emails]
adjusted_responses = ['NA' if response == "escalate" else response for response in all_emails]

escalator_outputs = pd.DataFrame({
    "Lead_Status": Lead_Status,
    "Agent Response": adjusted_responses
})

escalator_outputs.to_csv("escalator_outputs.csv", index=False)