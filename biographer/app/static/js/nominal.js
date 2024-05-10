document.getElementById('testNominal').addEventListener('change', function() {
    var binomOptions = document.getElementById('binomOptions');
    if (this.value === 'exact_test_of_goodness_of_fit') {
        binomOptions.style.display = 'block';
    } else {
        binomOptions.style.display = 'none';
    }
    
});