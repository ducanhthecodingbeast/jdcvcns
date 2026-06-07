document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const loadingOverlay = document.getElementById('loading-overlay');
  const resultsSection = document.getElementById('results-section');
  const jdGrid = document.getElementById('jd-grid');

  // Drag and Drop Events
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.remove('dragover');
    }, false);
  });

  // Handle file drop
  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  });

  // Handle file input selection
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    if (file.type !== 'application/pdf') {
      alert('Please upload a PDF file.');
      return;
    }

    // Simulate upload and processing
    loadingOverlay.classList.add('active');
    resultsSection.classList.remove('active');
    
    // Simulate API delay
    setTimeout(() => {
      loadingOverlay.classList.remove('active');
      displayResults();
    }, 2000);
  }

  function displayResults() {
    // Mock Data based on Enterprise IT roles
    const mockJDs = [
      {
        title: 'Senior Cloud Architect',
        department: 'Hybrid Cloud',
        location: 'New York, NY',
        score: '98% Match',
        description: 'Design and implement scalable cloud infrastructure using Kubernetes, Red Hat OpenShift, and AWS.'
      },
      {
        title: 'Data Scientist (AI/ML)',
        department: 'IBM Consulting',
        location: 'Remote',
        score: '92% Match',
        description: 'Develop enterprise-grade predictive models and natural language processing pipelines.'
      },
      {
        title: 'Full Stack Engineer',
        department: 'Software',
        location: 'San Jose, CA',
        score: '88% Match',
        description: 'Build robust web applications with React, Node.js, and DB2.'
      },
      {
        title: 'Cybersecurity Analyst',
        department: 'Security',
        location: 'Austin, TX',
        score: '81% Match',
        description: 'Monitor, detect, and respond to security threats across global enterprise networks.'
      }
    ];

    jdGrid.innerHTML = ''; // Clear previous

    mockJDs.forEach(jd => {
      const card = document.createElement('div');
      card.className = 'feature-card';
      card.innerHTML = `
        <div class="card-meta">
          <span>${jd.department}</span>
          <span>${jd.location}</span>
        </div>
        <div class="card-title">${jd.title}</div>
        <div class="body-sm" style="flex-grow: 1;">${jd.description}</div>
        <div class="card-footer">
          <span class="match-score">${jd.score}</span>
          <button class="btn-primary" style="padding: 8px 12px; font-size: 12px;">Apply Now</button>
        </div>
      `;
      jdGrid.appendChild(card);
    });

    resultsSection.classList.add('active');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
});
