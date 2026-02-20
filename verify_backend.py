import requests
import time
import sys

def verify():
    print("Waiting for backend to start...")
    url = "http://localhost:8000"
    for i in range(20):
        try:
            response = requests.get(f"{url}/")
            if response.status_code == 200:
                print("Backend is up!")
                break
        except Exception as e:
            pass
        time.sleep(1)
        print(f"Waiting... {i+1}/20")
    else:
        print("Backend failed to start within timeout.")
        return

    print("Testing /summarize endpoint...")
    text = "Artificial Intelligence is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans." * 5
    try:
        response = requests.post(f"{url}/summarize", json={"text": text})
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Summary Response:", response.json())
        else:
            print("Error Response:", response.text)
    except Exception as e:
        print("Request failed:", e)

if __name__ == "__main__":
    verify()
