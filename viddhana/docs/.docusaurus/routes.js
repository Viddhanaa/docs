import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/docs',
    component: ComponentCreator('/docs', '77e'),
    routes: [
      {
        path: '/docs/api-reference/json-rpc',
        component: ComponentCreator('/docs/api-reference/json-rpc', '145'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/docs/api-reference/overview',
        component: ComponentCreator('/docs/api-reference/overview', '224'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/docs/getting-started/installation',
        component: ComponentCreator('/docs/getting-started/installation', '381'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/docs/getting-started/quickstart',
        component: ComponentCreator('/docs/getting-started/quickstart', 'ec5'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/docs/intro',
        component: ComponentCreator('/docs/intro', 'f79'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/docs/sdks/javascript',
        component: ComponentCreator('/docs/sdks/javascript', '565'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/docs/sdks/python',
        component: ComponentCreator('/docs/sdks/python', 'b92'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/docs/smart-contracts/overview',
        component: ComponentCreator('/docs/smart-contracts/overview', '68f'),
        exact: true,
        sidebar: "docsSidebar"
      },
      {
        path: '/docs/smart-contracts/vdh-token',
        component: ComponentCreator('/docs/smart-contracts/vdh-token', '3fb'),
        exact: true,
        sidebar: "docsSidebar"
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '642'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
