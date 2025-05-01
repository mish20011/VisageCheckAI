import React,{useState , useEffect} from 'react';
import NoteContext from './NoteContext';

function NoteState(props) {
  const [globalUsername, setGlobalUsername] = useState(() => {
    return localStorage.getItem('globalUsername') || ''; // Initialize from localStorage
});
useEffect(() => {
  if (globalUsername) {
      localStorage.setItem('globalUsername', globalUsername);
      setLoggedIn(true); // Mark the user as logged in if username exists
  } else {
      localStorage.removeItem('globalUsername');
      setLoggedIn(false); // Mark the user as logged out if username is cleared
  }
}, [globalUsername]);
  const[loggedIn , setLoggedIn] = useState(false);
  return (
    <NoteContext.Provider  value={{ globalUsername, setGlobalUsername , loggedIn , setLoggedIn  }}>
        {props.children}
      </NoteContext.Provider>
  )
}

export default NoteState
