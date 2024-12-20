<template>
  <div class="App">
    <Header />
    <div class="main-content">
      <router-view></router-view>
    </div>
  </div>
</template>

<script setup>
import { onMounted, provide } from 'vue';
import { useAppStore } from './store/appStore';
import Header from './components/AppHeader.vue';

// Access appStore
const appStore = useAppStore();
const { state, addItem, updateItem, deleteItem, setConstants } = appStore;

// Provide appStore to child components
provide('appStore', {
  state,
  addItem,
  updateItem,
  deleteItem,
});

// Track loading states
let preferencesLoaded = false;
let constantsLoaded = false;

// Fetch data helper function
const fetchData = async (url) => {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.error(`Error fetching data from ${url}:`, err);
    throw err;
  }
};

// Load constants and store them in appStore
const loadConstants = async () => {
  try {
    const data = await fetchData('http://127.0.0.1:8000/get_constants');

    // Parse constants into a key-value format
    const parsedConstants = {};
    data.forEach((item) => {
      parsedConstants[item.name] = item.value;
    });

    // Set constants in the appStore
    setConstants(parsedConstants);

    constantsLoaded = true;
  } catch (err) {
    console.error('Error loading constants:', err);
  }
};

// Load preferences and store them in appStore
const loadPreferences = async (userId) => {
  try {
    const data = await fetchData(`http://127.0.0.1:8000/user_preferences/${userId}`);
    if (data) {
      data.forEach((item) => {
        let status_str = 'error';
        if (item.status === state.constants['STATUS_MORE'])
          status_str = 'more';
        if (item.status === state.constants['STATUS_LESS'])
          status_str = 'less';
        if (item.status === state.constants['STATUS_IGNORE'])
          status_str = 'ignore';

        addItem({
          name: item.name,
          id: item.game_id,
          rating: item.rating,
          status: status_str,
        });
      });
    }
    preferencesLoaded = true;
  } catch (err) {
    console.error('Error loading user preferences:', err);
  }
};

// Run setup functions on component mount
onMounted(() => {
  if (!constantsLoaded) {
    loadConstants();
  }

  if (!preferencesLoaded) {
    loadPreferences(4);
  }
});
</script>

<style scoped>
@import './styles/App.css';
</style>
