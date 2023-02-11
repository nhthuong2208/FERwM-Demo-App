function predictExprBtnClicked() {
    const image_display = document.querySelector("#image-display")
    const url = image_display.style.backgroundImage
    var model_chosen = document.getElementsByClassName("model-select")[0].value
    console.log(url.slice(4,url.length - 1))
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
         if (this.readyState == 4 && this.status == 200) {
            var result = JSON.parse(this.responseText)
            var positive_pred = parseFloat(result.positive).toFixed(2)
            var negative_pred = parseFloat(result.negative).toFixed(2)
            var neutral_pred = parseFloat(result.neutral).toFixed(2)
            var result_text = ""
            max_val = Math.max(result.positive, result.negative, result.neutral).toFixed(2)
            switch (max_val) {
                case positive_pred:
                    result_text = "positive"
                    break
                case negative_pred:
                    result_text = "negative"
                    break
                case neutral_pred:
                    result_text = "neutral"
                    break
            }
            document.getElementById("predict-result").innerHTML = `
                <div>Prediction</div>
                <div>Positive: ${positive_pred}, Negative: ${negative_pred}, Neutral: ${neutral_pred}</div>
                <div>Result: ${result_text}</div>
                <div>${model_chosen}</div>
            `
         }
    };
    xhttp.open("POST", "http://127.0.0.1:5000/predict", true);
    xhttp.setRequestHeader('Content-type', 'application/json');
    xhttp.send(JSON.stringify({"image": `${url.slice(5,url.length - 2)}`,"model":`${model_chosen}`}));
};