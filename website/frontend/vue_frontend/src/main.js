// main.js
import { createApp } from 'vue'
import App from './App.vue'
import router from './router/index.js'

import './styles/index.css'
import './styles/styles.css'

createApp(App)
  .use(router)
  .mount('#app')
