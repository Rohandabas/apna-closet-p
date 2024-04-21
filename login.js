let resdata = JSON.parse(localStorage.getItem("userData")) || [];
let form = document.getElementById("forms");

form.addEventListener("submit", (e)=>{
    e.preventDefault();
    let registered = false;
    let title = "";
    for(let i = 0; i < resdata.length; i++){
        if(resdata[i].email === form.email.value && resdata[i].pwd === form.psw.value){
            registered = true;
            title = resdata[i].name;
            break;
        }
    }

    // Retrieve the entered captcha code
    const enteredCaptcha = form.captcha.value.trim();
    // Retrieve the stored captcha code
    const storedCaptcha = localStorage.getItem("captcha");

    if (registered && enteredCaptcha === storedCaptcha) {
        localStorage.setItem("name", JSON.stringify(title));
        window.location = "main.html";
    } else {
        alert("Incorrect credentials or captcha. Please try again.");
        form.captcha.value = ''; // Reset captcha field
        generateCaptcha(); // Generate a new captcha
    }
});

// Function to generate and display a new captcha code
function generateCaptcha() {
    const captchaCode = Math.random().toString(36).substring(2, 8);
    localStorage.setItem("captcha", captchaCode); // Store captcha code
    const captchaContainer = document.getElementById('captchaContainer');
    captchaContainer.textContent = `Captcha: ${captchaCode}`;
}

// Generate captcha code initially
generateCaptcha();
