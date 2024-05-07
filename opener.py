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

model_name = "gemini-pro"
temperature = 0.5

llm = ChatGoogleGenerativeAI(model=model_name,temperature=temperature)

inputs = pd.DataFrame()

inputs["model_name"] = [model_name] * len(df)
inputs["temperature"] = [temperature] * len(df)

all_name = []
all_job_title = []
all_organization = []
all_project_title = []
all_looking_for = []

all_prompts = []

def generate_email(row):

    name = row['Name']
    job_title = row['Job Title']
    organization = row['Organization']
    project_title = row['Project Title']
    looking_for = row['Looking For']

    all_name.append(name)
    all_job_title.append(job_title)
    all_organization.append(organization)
    all_project_title.append(project_title)
    all_looking_for.append(looking_for)

    prompt = f"""You are responsible to be the Opener agent for the lead and will aim to establish connection to encourage further communication.

    You will generate a personalized email using the following details of the recipient:
    Name: {name}
    Job Title: {job_title}
    Organization: {organization}
    Project Title: {project_title}
    Looking For: {looking_for}

    The email should inquire about more details regarding the scope and budget they have in mind.

    The subject should be brief and to the point limited to only 30 characters.
    The body should be concise and limited to three paragraphs and should ask for more details about the project.
    The tone of the email should be human-like and professional.

    You will use my name as "Muhammad Ahmed Suhail" and my organization as "Antematter"

    Your output should be the subject of the email and the personalized email.
    
    
    """
    
    all_prompts.append(prompt)

    response = llm.invoke(prompt).content
    return response

all_emails = df.apply(generate_email, axis=1)

inputs["name"] = all_name
inputs["job_title"] = all_job_title
inputs["organization"] = all_organization
inputs["project_title"] = all_project_title
inputs["looking_for"] = all_looking_for
inputs["prompt"] = all_prompts

inputs.to_csv("opener_inputs.csv", index=False)

all_subject = []
all_body = []

for email_text in all_emails:

    parts = email_text.split('\n\n', 1)
    subject_line = parts[0].strip()
    subject = subject_line.replace('**Subject:', '').replace('**', '').strip()
    body = parts[1].strip() if len(parts) > 1 else ""

    all_subject.append(subject)
    all_body.append(body)

outputs = pd.DataFrame()

outputs["Subject"] = all_subject
outputs["Body"] = all_body

outputs.to_csv("opener_outputs.csv", index=False)