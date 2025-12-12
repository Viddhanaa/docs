// @ts-check
const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'VIDDHANA Developer Hub',
  tagline: 'Build on VIDDHANA: The Operating System for Wealth',
  favicon: 'img/logo.png',

  url: 'https://docs.viddhana.com',
  baseUrl: '/',

  organizationName: 'viddhana',
  projectName: 'docs',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/viddhana/docs/edit/main/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-GvrOXuhMATgEsSwCs4smOFZETl1RdQhRs2Jl1x1qVjGw45p3R9kqOlwj8u3h/L8v',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/social-card.jpg',

      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },

      navbar: {
        title: 'VIDDHANA',
        logo: {
          alt: 'VIDDHANA Logo',
          src: 'img/logo.png',
          srcDark: 'img/logo.png',
          height: 32,
          width: 32,
          style: { borderRadius: '4px' },
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Docs',
          },
          {
            href: '/docs/api-reference/overview',
            label: 'API',
            position: 'left',
          },
          {
            href: '/docs/sdks/javascript',
            label: 'SDKs',
            position: 'left',
          },
          {
            href: 'https://github.com/viddhana',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://discord.gg/viddhana',
            label: 'Discord',
            position: 'right',
          },
        ],
      },

      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              { label: 'Getting Started', to: '/docs/intro' },
              { label: 'Smart Contracts', to: '/docs/smart-contracts/overview' },
              { label: 'API Reference', to: '/docs/api-reference/overview' },
            ],
          },
          {
            title: 'Community',
            items: [
              { label: 'Discord', href: 'https://discord.gg/viddhana' },
              { label: 'Twitter', href: 'https://twitter.com/viddhana' },
              { label: 'Telegram', href: 'https://t.me/viddhana' },
            ],
          },
          {
            title: 'Resources',
            items: [
              { label: 'GitHub', href: 'https://github.com/viddhana' },
              { label: 'Whitepaper', href: 'https://viddhana.com/whitepaper.pdf' },
            ],
          },
        ],
        copyright: `Copyright ${new Date().getFullYear()} VIDDHANA. Built with Docusaurus.`,
      },

      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['solidity', 'python', 'go', 'bash', 'json'],
      },
    }),
};

module.exports = config;
