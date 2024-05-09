document.getElementById('testOneMeasurement').addEventListener('change', function() {
    var sampleMeans = document.getElementById('sampleMeans');
    if (this.value === 'one_sample_t_test') {
        sampleMeans.style.display = 'block';
    } else {
        sampleMeans.style.display = 'none';
    }
});