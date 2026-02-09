document.addEventListener('DOMContentLoaded', () => {
    initFileUpload();
    initAnimations();
});

function initFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const videoInput = document.getElementById('videoInput');
    const fileName = document.getElementById('fileName');

    if (!uploadArea || !videoInput) return;

    uploadArea.addEventListener('click', () => videoInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            videoInput.files = e.dataTransfer.files;
            updateFileName();
        }
    });

    videoInput.addEventListener('change', updateFileName);

    function updateFileName() {
        if (videoInput.files.length > 0) {
            const file = videoInput.files[0];
            const size = formatFileSize(file.size);
            fileName.textContent = `${file.name} (${size})`;
            uploadArea.classList.add('has-file');
        }
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function initAnimations() {
    const fadeElements = document.querySelectorAll('.hero-content, .upload-card');
    fadeElements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, 100 + (index * 150));
    });
}
