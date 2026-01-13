document.getElementByid("predictForm").addEventListener("submit",async function(e){
    e.preventDefault();

    const data={
        marital_status:document.getElementById("marital_status").value,
        application_mode:document.getElementById("application_mode").value,
        application_order:parseInt(document.getElementById("application_order").value),
        course:document.getElementById("course").value,
        daytime_evening_attendance:document.getElementById("daytime_evening_attendance").value,
        previous_qualification:document.getElementById("previous_qualification").value,
        mothers_qualification:document.getElementById("mothers_qualification").value,
        fathers_occupation:document.getElementById("fathers_occupation").value,
        displaced:document.getElementById("displaced").value,
        educational_special_needs:document.getElementById("educational_special_needs").value,
        debtor:document.getElementById("debtor").value,
        tuition_fees_up_to_date:document.getElementById("tuition_fees_up_to_date").value,
        gender:document.getElementById("gender").value,
        scholarship_holder:document.getElementById("scholarship_holder").value,
        age_at_enrollment:parseInt(document.getElementById("age_at_enrollment").value),
        international:document.getElementById("international").value,
        Curricular_units_1st_sem_credited:parseFloat(document.getElementById("Curricular_units_1st_sem_credited").values),
        Curricular_units_1st_sem_enrolled:parseFloat(document.getElementById("Curricular_units_1st_sem_enrolled").values),
        Curricular_units_1st_sem_evaluations:parseFloat(document.getElementById("Curricular_units_1st_sem_evaluations").values),
        Curricular_units_1st_sem_approved:parseFloat(document.getElementById("Curricular_units_1st_sem_approved").values),
        Curricular_units_1st_sem_grade:parseFloat(document.getElementById("Curricular_units_1st_sem_grade").values),
        Curricular_units_1st_sem_without_evaluations:parseFloat(document.getElementById("Curricular_units_1st_sem_without_evaluations").values),
        Curricular_units_2nd_sem_credited:parseFloat(document.getElementById("Curricular_units_2nd_sem_credited").values),
        Curricular_units_2nd_sem_enrolled:parseFloat(document.getElementById("Curricular_units_2nd_sem_enrolled").values),
        Curricular_units_2nd_sem_evaluations:parseFloat(document.getElementById("Curricular_units_2nd_sem_evaluations").values),
        Curricular_units_2nd_sem_approved:parseFloat(document.getElementById("Curricular_units_2nd_sem_approved").values),
        Curricular_units_2nd_sem_grade:parseFloat(document.getElementById("Curricular_units_2nd_sem_grade").values),
        Curricular_units_2nd_sem_without_evaluations:parseFloat(document.getElementById("Curricular_units_2nd_sem_without_evaluations").values),

    };

    const response=await fetch("/predict",{
        method:"POST",
        hedaers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify(data)

    });

    const result=await response.json();

    document.getElementById("result").innerHTML='Risk: ${result.prediction}<br>Probability:${result.dropout_probability}';


    

});