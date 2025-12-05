import requests
import os

API_URL = "http://localhost:8000"

def test_api():
    print("Testing API...")
    
    # 1. List Notebooks (should be empty or existing)
    try:
        response = requests.get(f"{API_URL}/notebooks")
        if response.status_code == 200:
            print("List Notebooks: OK", response.json())
        else:
            print("List Notebooks: Failed", response.status_code)
            return
    except requests.exceptions.ConnectionError:
        print("Backend not running. Please start backend first.")
        return

    # 2. Create Notebook
    notebook_name = "TestNotebook_Verification"
    response = requests.post(f"{API_URL}/notebooks", json={"name": notebook_name})
    if response.status_code == 200:
        print("Create Notebook: OK")
    else:
        print("Create Notebook: Failed", response.text)

    # 3. List again
    response = requests.get(f"{API_URL}/notebooks")
    print("List Notebooks (After Create):", response.json())

    # 4. Delete Notebook
    response = requests.delete(f"{API_URL}/notebooks/{notebook_name}")
    if response.status_code == 200:
        print("Delete Notebook: OK")
    else:
        print("Delete Notebook: Failed", response.text)

if __name__ == "__main__":
    test_api()
