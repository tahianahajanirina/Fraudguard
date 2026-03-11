"""Load test for FraudGuard /predict endpoint."""

from locust import HttpUser, between, task

# Real transaction from the creditcard.csv dataset (non-fraudulent)
NORMAL_TRANSACTION = {
    "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
    "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986,
    "V9": 0.3637, "V10": 0.0907, "V11": -0.5515, "V12": -0.6178,
    "V13": -0.9913, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704,
    "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210,
    "Amount": 149.62,
}

# Suspicious transaction (high-risk pattern)
FRAUD_TRANSACTION = {
    "V1": -3.0435, "V2": -3.1574, "V3": 1.0888, "V4": 2.2886,
    "V5": 1.3597, "V6": -1.0648, "V7": 0.3252, "V8": -0.0677,
    "V9": -0.2708, "V10": -0.8385, "V11": -0.4143, "V12": -0.5032,
    "V13": 0.6768, "V14": -5.4041, "V15": -0.0728, "V16": -0.4608,
    "V17": -0.2181, "V18": -0.0284, "V19": -0.3625, "V20": -0.0560,
    "V21": -0.2028, "V22": 0.0783, "V23": -0.0597, "V24": -0.3111,
    "V25": -0.3280, "V26": -0.2972, "V27": -0.0550, "V28": -0.0597,
    "Amount": 1.00,
}


class FraudApiUser(HttpUser):
    """Simulated user calling the FraudGuard API."""

    wait_time = between(0.5, 2)

    @task(3)
    def predict_normal(self):
        """Send a normal transaction (most common scenario)."""
        self.client.post("/predict", json=NORMAL_TRANSACTION)

    @task(1)
    def predict_fraud(self):
        """Send a suspicious transaction."""
        self.client.post("/predict", json=FRAUD_TRANSACTION)

    @task(1)
    def health_check(self):
        """Check API health."""
        self.client.get("/health")