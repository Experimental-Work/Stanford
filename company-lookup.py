import openai
import os
from ...dotenv import load_dotenv


# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Set up OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No OpenAI API key found. Make sure it's set in your .env file.")
client = openai.OpenAI(api_key=openai_api_key)


def chat(message, system_content):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": message},
        ]
    )
    return response.choices[0].message.content


def main():
    # Begin script with identification of company
    company_name = input("What company do you want to learn about? ")
    company_url = input("What is their URL (to make sure we're talking about the same company)? ")

    # Ask the user what information they want to retrieve
    print("\nWhat would you like to know about the company?")
    print("1. Main product(s)/lines")
    print("2. Industry/sectors they operate in")
    print("3. Current level(s) of competition")
    print("4. Expected future prospects over the next 5 years")
    choice = input("Enter your choice (1-4): ")

    # Assemble the query based on user's choices
    if choice == '1':
        query = f"What are the main product(s) or product lines of {company_name} (URL: {company_url})?"
    elif choice == '2':
        query = f"In which industry/sectors does {company_name} (URL: {company_url}) operate?"
    elif choice == '3':
        query = f"What is the current level of competition for {company_name} (URL: {company_url})?"
    elif choice == '4':
        query = f"What are the expected future prospects over the next 5 years for {company_name} (URL: {company_url})?"
    else:
        print("Invalid choice. Exiting.")
        return

    # Get information from OpenAI
    system_content = "You are a helpful assistant that provides concise and accurate information about companies."
    response = chat(query, system_content)

    # Print the response
    print("\nHere's the information you requested:")
    print(response)


if __name__ == "__main__":
    main()
