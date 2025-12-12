# Logo & Assets

## Files

- **logo.png** - Logo chính mới (ACTIVE - 36,567 bytes)
- **logo.svg.old** - Logo copy (36,567 bytes) - cùng với logo.png
- **favicon.ico** - Favicon cũ (backup - 582 bytes)
- **favicon.ico.old** - Backup của favicon gốc
- **social-card.jpg** - Logo mới (36,567 bytes) - thay thế hoàn toàn
- **social-card-backup.jpg** - Logo mới (36,567 bytes) - thay thế hoàn toàn

**Note**: Tất cả image files (trừ favicon.ico cũ) hiện đều sử dụng logo.png mới!
Favicon được config sử dụng logo.png trong docusaurus.config.js

## Sử dụng

Logo hiện tại được sử dụng trong:
- Navbar (docusaurus.config.js)
- Responsive với kích thước 32x32px
- Border radius 4px để bo góc nhẹ
- Hover effect scale 1.05x

## Tùy Chỉnh

Để thay đổi kích thước logo, edit trong `docusaurus.config.js`:

```javascript
logo: {
  alt: 'VIDDHANA Logo',
  src: 'img/logo.png',
  srcDark: 'img/logo.png',
  height: 32,  // Thay đổi chiều cao
  width: 32,   // Thay đổi chiều rộng
  style: { borderRadius: '4px' },
},
```

## CSS Styling

Custom styling trong `src/css/custom.css`:

```css
.navbar__logo {
  height: 32px !important;
  width: 32px !important;
  border-radius: 4px;
  object-fit: contain;
  transition: transform 0.2s ease;
}

.navbar__logo:hover {
  transform: scale(1.05);
}
```

