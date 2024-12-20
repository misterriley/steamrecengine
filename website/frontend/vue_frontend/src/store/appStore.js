import { reactive } from 'vue';

const state = reactive({
  selectedItems: [],
  constants: {}, // Store for dynamically loaded constants
});

function addItem(item) {
  state.selectedItems.push(item);
}

function updateItem(index, updatedFields) {
  const items = state.selectedItems;
  items[index] = { ...items[index], ...updatedFields };
}

function deleteItem(index) {
  state.selectedItems.splice(index, 1);
}

// Function to set constants dynamically
function setConstants(loadedConstants) {
  state.constants = loadedConstants;
}

export function useAppStore() {
  return {
    state,
    addItem,
    updateItem,
    deleteItem,
    setConstants
  };
}
