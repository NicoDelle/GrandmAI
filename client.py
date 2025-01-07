import requests

def send_prompt(prompt):
    url = "http://localhost:5000/process"
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt}
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to get response from server"}

print("Welcome to grandmAI!")
while True:
    prompt = input(">>> ")
    response = send_prompt(prompt)
    print(response['response'])