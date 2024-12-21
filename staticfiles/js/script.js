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
