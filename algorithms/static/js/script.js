document.addEventListener("DOMContentLoaded", function () {
    const buttons = document.querySelectorAll("button[data-confirm]");

    buttons.forEach(button => {
        button.addEventListener("click", function (e) {
            const confirmMessage = button.getAttribute("data-confirm");
            if (!confirm(confirmMessage)) {
                e.preventDefault();
            }
        });
    });
});

async function startAlgorithm() {
    const functionName = document.getElementById("functionSelect").value;
    const dimensions = document.getElementById("dimensions").value;
    const populationSize = document.getElementById("populationSize").value;
    const maxIterations = document.getElementById("maxIterations").value;
    document.getElementById("pauseButton").style.display = "inline";

    try {
        const response = await fetch(algorithmData.url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": algorithmData.csrfToken
            },
            body: JSON.stringify({
                algorithm: algorithmData.algorithmName,
                function: functionName,
                dimensions: dimensions,
                population_size: populationSize,
                max_iterations: maxIterations
            })
        });

        const data = await response.json();

        if (response.ok) {
            const resultLink = resultUrl;
            console.log(resultLink);
            document.getElementById("results").innerHTML = `
                <p>Algorithm executed successfully!</p>
                <a href="${resultLink}">View Results</a>
            `;
        } else {
            document.getElementById("results").innerHTML = `<p>Error: ${data.error || "Unknown error occurred"}</p>`;
        }
    } catch (error) {
        document.getElementById("results").innerHTML = `<p>Error: ${error.message}</p>`;
    }
}


async function pauseAlgorithm() {
    try {
        const response = await fetch("/pause/", { method: "POST" });
        const data = await response.json();
        if (response.ok) {
            document.getElementById("pauseButton").style.display = "none";
            document.getElementById("resumeButton").style.display = "inline";
            console.log(data.message);
        }
    } catch (error) {
        console.error("Error pausing algorithm:", error);
    }
}

async function resumeAlgorithm() {
    try {
        const response = await fetch("/resume/", { method: "POST" });
        const data = await response.json();
        if (response.ok) {
            document.getElementById("resumeButton").style.display = "none";
            document.getElementById("pauseButton").style.display = "inline";
            console.log(data.message);
        }
    } catch (error) {
        console.error("Error resuming algorithm:", error);
    }
}
