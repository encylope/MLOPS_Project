# User Manual — FraudGuard Portal
## Non-Technical User Guide

---

## What is FraudGuard?

FraudGuard is a web application that analyzes credit card transactions and tells you
whether they are likely **legitimate** or **fraudulent**. It uses a machine learning
model trained on real transaction data to give you an instant risk assessment.

---

## Accessing the Application

Open your web browser and go to: **http://localhost:3000**

You will see three sections in the top navigation bar:
- **Predict** — Analyze a transaction
- **Pipeline** — View the machine learning pipeline status
- **Dashboard** — See live performance metrics

---

## How to Analyze a Transaction (Predict page)

### Step 1 — Try a sample transaction (easiest way to start)

Click one of these buttons at the top of the form:
- **"Load legitimate"** — fills in a sample normal transaction
- **"Load fraud"** — fills in a sample fraudulent transaction

### Step 2 — Fill in the transaction details

The form has 30 fields:
- **V1 to V28** — These are security-encoded card features (numbers that describe
  the transaction in an anonymized way for privacy)
- **Amount (USD)** — The dollar value of the transaction
- **Time** — How many seconds since the first transaction of the day

> **Tip:** In real use, these values come directly from your payment processing system.
> You do not need to know what V1–V28 mean — the AI handles that.

### Step 3 — Click "Analyze Transaction"

The button will show "Analyzing..." while the AI processes your input (usually under a second).

### Step 4 — Read the result

The result panel shows:

| Field | What it means |
|-------|--------------|
| **LEGITIMATE / FRAUD** | The AI's verdict |
| **Fraud probability bar** | Higher % = more suspicious (0% = definitely safe, 100% = definitely fraud) |
| **Risk level** | LOW (under 30%), MEDIUM (30–70%), HIGH (over 70%) |
| **Amount** | The transaction amount you entered |
| **Model version** | Which version of the AI was used |
| **Latency** | How fast the AI responded (in milliseconds) |
| **Transaction ID** | A unique reference number for this analysis |

---

## Interpreting Results

- **LOW risk (green):** Transaction looks normal. No immediate action needed.
- **MEDIUM risk (yellow):** Unusual patterns detected. Consider manual review.
- **HIGH risk (red):** Strong fraud signals. Consider blocking or contacting the cardholder.

> **Important:** The AI provides a probability score, not a guaranteed verdict. Always
> use your judgment alongside the result, especially for borderline cases.

---

## Pipeline Page

This page shows the stages the AI went through to learn from data:

1. **Data Validation** — Checks the raw transaction data for errors
2. **Preprocessing** — Cleans and scales the data
3. **Feature Engineering** — Prepares the features the AI learns from
4. **Train/Val/Test Split** — Divides data for training and testing
5. **Model Training** — The AI learns patterns from historical transactions
6. **Evaluation** — Tests how accurate the AI is

You can also see links to technical monitoring tools used by the development team.

---

## Dashboard Page

Shows live performance information:

- **Total requests** — How many transactions have been analyzed
- **Fraud detections** — How many were flagged as fraud
- **Fraud rate** — What percentage of transactions were flagged
- **Avg latency** — How fast the system is responding

The charts update automatically every 5 seconds.

---

## Frequently Asked Questions

**Q: What if I see "Backend unavailable"?**  
A: The API server may still be starting. Wait 30 seconds and refresh the page.

**Q: What if the Analyze button is greyed out?**  
A: Make sure all 30 fields (V1–V28, Amount, Time) are filled in. Use "Load legitimate" to populate them.

**Q: Can I analyze multiple transactions at once?**  
A: The technical team can use the batch API endpoint. The web portal currently supports one transaction at a time.

**Q: Is my transaction data stored?**  
A: No. Transactions are processed in memory and are not saved to any database.

---

## Getting Help

If the application is not working, contact your system administrator and provide:
1. The Transaction ID from the result (if available)
2. A screenshot of any error message
3. The time the error occurred
