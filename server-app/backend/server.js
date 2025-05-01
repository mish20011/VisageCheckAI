import mongoose from "mongoose";
import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import {chatAuthData , myMessages} from "./dataStorage.js"; 
const uri = "mongodb+srv://MidnightGamer:Tester123@cluster0.wqmrn.mongodb.net/ChatSpace?retryWrites=true&w=majority&appName=Cluster0";

const app = express();
const PORT = process.env.PORT || 8001;
app.use(cors());
app.use(bodyParser.json());

const connectDb = async () => {
  try {
    await mongoose.connect(uri); // Removed deprecated options
    console.log("Connected to MongoDB Atlas");
  } catch (error) {
    console.error("Could not connect to MongoDB Atlas", error);
    process.exit(1);
  }
};

// Signup Route
app.post("/signup", async (req, res) => {
  const { username, password} = req.body;

  if (!username || !password ) {
    return res.status(400).json({ error: "Username, password are required." });
  }

  try {
    // Check if the username already exists
    const existingUser = await chatAuthData.findOne({ username });
    if (existingUser) {
      return res.status(409).json({ error: "Username already exists." });
    }


    const newUser = new chatAuthData({ username, password: password });
    const savedUser = await newUser.save();

    res.status(201).json({ message: "User signed up successfully.", user: savedUser });
  } catch (error) {
    console.error("Error signing up:", error);
    res.status(500).json({ error: "Error signing up." });
  }
});

// Login Route
app.post("/login", async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ error: "Username and password are required." });
  }

  try {
    // Find user by username
    const user = await chatAuthData.findOne({ username });
    if (!user) {
      return res.status(404).json({ error: "User not found." });
    }

    // Compare the password
    const isMatch = (password === user.password);
    if (!isMatch) {
      return res.status(401).json({ error: "Invalid credentials." });
    }

    res.status(200).json({ message: "Login successful.", user });
  } catch (error) {
    console.error("Error logging in:", error);
    res.status(500).json({ error: "Error logging in." });
  }
});
app.post("/saveMessage", async (req, response) => {
  const { username, query, imageBase64, res, doctors,desc } = req.body;

  if (!username || !res) {
    return response.status(400).json({ error: "Username and response are required." });
  }

  try {
    const newMessage = new myMessages({ username, query, imageBase64, res, doctors,desc });
    await newMessage.save();
    response.status(201).json({ message: "Message saved successfully." });
  } catch (error) {
    console.error("Error saving message:", error);
    response.status(500).json({ error: "Failed to save message." });
  }
});
app.post('/getMessages', async (req, res) => {
  const { username } = req.body;

  // Validate the input
  if (!username) {
    return res.status(400).json({
      error: "Username is required.",
    });
  }

  try {
    // Fetch all messages for the given username
    const userMessages = await myMessages.find({ username });

    // Check if messages exist
    if (userMessages.length === 0) {
      return res.status(404).json({
        message: "No messages found for this username.",
      });
    }

    // Return the messages
    res.status(200).json({
      message: "Messages retrieved successfully.",
      myData: userMessages,
    });
  } catch (error) {
    console.error("Error retrieving messages:", error);
    res.status(500).json({
      error: "Failed to retrieve messages.",
    });
  }
});

app.listen(PORT, async () => {
    await connectDb();
    console.log(`Server is running on http://localhost:${PORT}`);
  });
  