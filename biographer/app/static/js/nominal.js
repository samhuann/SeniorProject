document.getElementById('testNominal').addEventListener('change', function() {
    var binomOptions = document.getElementById('binomOptions');
    if (this.value === 'exact_test_of_goodness_of_fit') {
        binomOptions.style.display = 'block';
    } else {
        binomOptions.style.display = 'none';
    }
    var chigoodDisplay = document.getElementById('chigoodDisplay');
    if (this.value === 'chi_square_test_of_goodness_of_fit') {
        chigoodDisplay.style.display = 'block';
    } else {
        chigoodDisplay.style.display = 'none';
    }
    if (this.value === 'chi_square_test_of_independence') {
        chiindDisplay.style.display = 'block';
    } else {
        chiindDisplay.style.display = 'none';
    }
    if (this.value === 'fishers_exact_test') {
        fisherdisplay.style.display = 'block';
    } else {
        fisherdisplay.style.display = 'none';
    }
    
});