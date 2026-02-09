document.addEventListener('DOMContentLoaded', () => {
    initMobileMenu();
    initNavSections();
    initActiveState();
});

function initMobileMenu() {
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.querySelector('.sidebar');
    
    if (!menuToggle || !sidebar) return;

    menuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 1024) {
            if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        }
    });
}

function initNavSections() {
    const sectionTitles = document.querySelectorAll('.nav-section-title');
    
    sectionTitles.forEach(title => {
        title.addEventListener('click', () => {
            const section = title.parentElement;
            const dropdown = section.querySelector('.nav-dropdown');
            
            section.classList.toggle('collapsed');
            
            if (dropdown) {
                if (section.classList.contains('collapsed')) {
                    dropdown.style.maxHeight = '0';
                    dropdown.style.opacity = '0';
                } else {
                    dropdown.style.maxHeight = dropdown.scrollHeight + 'px';
                    dropdown.style.opacity = '1';
                }
            }
        });
    });

    document.querySelectorAll('.nav-dropdown').forEach(dropdown => {
        dropdown.style.maxHeight = dropdown.scrollHeight + 'px';
        dropdown.style.opacity = '1';
        dropdown.style.transition = 'max-height 0.3s ease, opacity 0.3s ease';
    });
}

function initActiveState() {
    const currentPath = window.location.pathname;
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        const href = item.getAttribute('href');
        if (href && currentPath.includes(href)) {
            item.classList.add('active');
        }
    });
}

document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', () => {
        const btn = form.querySelector('button[type="submit"]');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        }
    });
});
