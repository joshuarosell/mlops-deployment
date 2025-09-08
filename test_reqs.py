import requests
import numpy as np
import time
import random 

BASE_URL = "http://localhost:8080"

def test_health_check():
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check status: {response.status_code}")
    print(f"Response: {response.json()}")

def send_prediction_request():
    data = {
        "amt": round(random.uniform(1, 1000), 2),
        "hour": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "category": random.choice(["grocery", "entertainment", "travel", "food"]),
        "state": random.choice(["CA", "NY", "TX", "FL"])
    }
    response = requests.post(f"{BASE_URL}/predict/", json=data)
    print(f"Prediction request status: {response.status_code}")
    print(f"Response: {response.json()}")

def simulate_user_traffic(num_requests=10, delay=1):
    print(f"Simulating {num_requests} user requests with {delay} second delay between requests")
    for i in range(num_requests):
        print(f"\nRequest {i+1}:")
        send_prediction_request()
        time.sleep(delay)

if __name__ == "__main__":
    test_health_check()
    print("\nStarting user traffic simulation...")
    simulate_user_traffic()