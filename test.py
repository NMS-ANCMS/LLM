import requests
%store -r ngrokURL

# Define the data to send in the POST request
data = {
    "inputs": '''
I found a problem in my connection to cisco switch. what should I do?
''',
    "parameters": {"temperature": 0,
                   "max_tokens": 200}
}

# Send the POST request
response = requests.post("/api/v1/show_user/${res.data.user_id}" + "/generate/", json=data)

# Check the response
if response.status_code == 200:
    result = response.json()
    print("Generated Text:\n", data["inputs"], result["generated_text"].strip())
else:
    print("Request failed with status code:", response.status_code)

