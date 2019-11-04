(function() {
    'use strict';
    window.addEventListener('load', function() {
        var forms = document.getElementsByClassName('needs-validation');
        var validation = Array.prototype.filter.call(forms, function(form) {
            form.addEventListener('submit', function(event) {
                if (form.checkValidity() === false) {
                    event.preventDefault();
                    event.stopPropagation();
                }
                form.classList.add('was-validated');
            }, false);
        });
    }, false);
})();

$("#theForm").submit(function(e) {

    if ($("#theForm")[0].checkValidity()) {
        e.preventDefault();

        var form = $(this);
        var url = form.attr('action');

        $.ajax({
            type: "GET",
            url: url,
            data: form.serialize(),
            success: function(data) {
                if (data == 'ERRO2') {
                    $("#successdiv").css("display", "none");
                    $("#errordiv").css("display", "block");
                } else if ((data.prediction == undefined) && (data.rmse == undefined)) {
                    document.getElementById("erromsg").innerHTML = ""
                    $("#successdiv").css("display", "none");
                    $("#errordiv").css("display", "block");
                    for (x in data) {
                        document.getElementById("erromsg").innerHTML += x + " --- " + data[x] + "<br>";
                    }
                } else {
                    $("#errordiv").css("display", "none");
                    $("#successdiv").css("display", "block");
                    prediction = "House Price Prediction: " + data.prediction
                    rmse = "RMSE: " + data.rmse
                    $("#hprice").html(prediction);
                    $("#rmse").html(rmse);
                }
            }
        });
    } else {
        $("#successdiv").css("display", "none");
        $("#errordiv").css("display", "block");
        $("#errordiv").html("Form validation failed - Wrong input parameters");
    };


});