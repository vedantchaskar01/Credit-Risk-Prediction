// 1. Grab the HTML elements we want to interact with
const form = document.getElementById('predictionForm');
const resultCard = document.getElementById('resultCard');
const probabilityText = document.getElementById('probabilityText');
const resultMessage = document.getElementById('resultMessage');

// 2. Listen for the user clicking the "Analyze Risk" button
form.addEventListener('submit', async function(e) {
    e.preventDefault(); // Stop the page from reloading!

    // 3. Gather the data from the form
    const applicantData = {
        age: parseInt(document.getElementById('age').value),
        income: parseFloat(document.getElementById('income').value),
        credit_score: parseInt(document.getElementById('credit_score').value),
        employment_type: document.getElementById('employment_type').value
    };

    try {
        // 4. Send the order to our Waiter (FastAPI)
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(applicantData)
        });

        const result = await response.json();

        // 5. Update the UI with the prediction!
        resultCard.classList.remove('hidden'); // Make the card visible
        
        // Convert probability (e.g. 0.85) to a percentage (85%)
        probabilityText.textContent = (result.probability_of_default * 100).toFixed(1) + '%';
        resultMessage.textContent = result.message;

        // 6. Change colors based on risk
        if (result.reject_loan) {
            resultMessage.className = 'result-message risk-high'; // Red
            document.querySelector('.probability-circle').style.borderColor = '#ef4444';
        } else {
            resultMessage.className = 'result-message risk-low'; // Green
            document.querySelector('.probability-circle').style.borderColor = '#10b981';
        }

    } catch (error) {
        alert("Error connecting to the API! Make sure your FastAPI server is running.");
    }
});
