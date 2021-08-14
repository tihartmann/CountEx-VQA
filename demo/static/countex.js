$(document).ready(function(){
    console.log("CountEx-VQA started.");
    clear_data();

    $(".custom-file-input").on("change", function() {
        var fileName = $(this).val().split("\\").pop();
        $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
        $(inputFile).removeClass('is-invalid');
        $('.custom-file-label').removeClass('custom-file-label-highlight');
    });

    $(function() {
        $('#upload-file-btn').click(function() {
            var form_data = new FormData($('#upload-file')[0]);
            call_mexpub(form_data); 
            return false;
        });

        function call_mexpub(pdf) {
            var filename = pdf.entries().next().value[1].name
            console.log("Calling mexpub with " + filename)
            if(filename == ''){
                $(inputFile).addClass('is-invalid');
                $('.custom-file-label').addClass('custom-file-label-highlight');
                return;
            }else{
                $(inputFile).removeClass('is-invalid');
                $('.custom-file-label').removeClass('custom-file-label-highlight');
            }
            clear_data(); //clear data before call
            $.ajax({
                type: 'POST',
                url: '/uploadpdf',
                data: pdf,
                contentType: false,
                cache: false,
                processData: false,
                beforeSend:function(){
                    $('nav').addClass('blurBackground');
                    $('.geekyContainer').addClass('blurBackground');
                    $(footerRow).addClass('blurBackground');
                    $(loader).show();
                    $('#inputFile').prop("disabled", true);
                    $('#upload-file-btn').prop("disabled", true);
                    $("#json-tab").css("pointer-events", "none");
                    $("#plaintext-tab").css("pointer-events", "none");
                    $("#geekyLink").css("pointer-events", "none");
                },
                success: function(data) {
                    //console.log(data);
                    if ("output" in data) {
                        console.log("Setting data")
                        set_data(data)
                    } else {
                        $("#displayfilename").html("No metadata for " + filename + ". Please validate the PDF file.");
                    }
                    $(loader).hide();
                    $('nav').removeClass('blurBackground');
                    $('.geekyContainer').removeClass('blurBackground');
                    $(footerRow).removeClass('blurBackground');
                    $('#inputFile').prop("disabled", false);
                    $('#upload-file-btn').prop("disabled", false);
                    $("#json-tab").css("pointer-events", "auto");
                    $("#plaintext-tab").css("pointer-events", "auto");
                    $("#geekyLink").css("pointer-events", "auto");
                    setTimeout(function(){ 
                        $.ajax({
                            url: '/deletefile/'+filename
                        })
                    }, 10000);
                    
                },
            });
        };
    });


    function clear_data() {
        $("#img").attr("src", "/static/placeholder.png")
        //$("#displayfilename").html("");
        //$("#file_name").html("");
        //$("#name").html("");
        $("#author").hide();
        $("#journal").hide();
        $("#date").hide();
        $("#doi").hide();
        $("#email").hide();
        $("#affiliation").hide();
        $("#address").hide();
        $("#abstract").hide();
        $("#titledata").hide();

    }

    function set_data(data) {
        console.log(data["image_path"])
        $("#img").attr("src", data["image_path"])
        $("#output").html(JSON.stringify(data["output"], null, 5));
        if ("title" in data["output"]) {
            $("#titledata").show();
            $("#title-prediction").text(data["output"]["title"])
        };
        if ("author" in data["output"]) {
            $("#author").show();
            $("#author-prediction").text(data["output"]["author"]);
        };
        if ("affiliation" in data["output"]) {
            $("#affiliation").show();
            $("#affiliation-prediction").text(data["output"]["affiliation"]);
        };
        if ("address" in data["output"]) {
            $("#address").show();
            $("#address-prediction").text(data["output"]["address"]);
        };
        if ("journal" in data["output"]) {
            $("#journal").show();
            $("#journal-prediction").text(data["output"]["journal"]);
        };
        if ("doi" in data["output"]) {
            $("#doi").show();
            $("#doi-prediction").text(data["output"]["doi"]);
        };
        if ("abstract" in data["output"]) {
            $("#abstract").show();
            $("#abstract-prediction").text(data["output"]["abstract"]);
        };
        if ("email" in data["output"]) {
            $("#email").show();
            $("#email-prediction").text(data["output"]["email"]);
        };
        if ("date" in data["output"]) {
            $("#date").show();
            $("#date-prediction").text(data["output"]["date"]);
        };
    };
});