import mongoose from "mongoose";
import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import {chatAuthData , myMessages} from "./dataStorage.js"; 

// const uri = "mongodb://localhost:27017/ChatSpace";
const uri = "mongodb://127.0.0.1:27017/ChatSpace";

const app = express();
const PORT = process.env.PORT || 8001;
app.use(cors());

// ⬇️ Set file size limit to 10mb or more
app.use(bodyParser.json({ limit: '20mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '20mb' }));


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
  const { username, password } = req.body;

  try {
    const existingUser = await chatAuthData.findOne({ username });
    if (existingUser) {
      return res.status(400).json({ message: "User already exists" });
    }

    const newUser = new chatAuthData({ username, password });
    await newUser.save();
    console.log("✅ Signup successful for user:", username);
    res.status(200).json({ message: "Signup successful" });

  } catch (error) {
    console.error("❌ Signup error:", error);
    res.status(500).json({ message: "Internal server error during signup" });
  }
});

// Login Route
app.post("/login", async (req, res) => {
  const { username, password } = req.body;

  try {
    const existingUser = await chatAuthData.findOne({ username });
    if (!existingUser) {
      return res.status(401).json({ message: "User not found" });
    }

    if (existingUser.password !== password) {
      return res.status(401).json({ message: "Incorrect password" });
    }

    console.log("✅ Login successful for user:", username);
    res.status(200).json({ message: "Login successful" });

  } catch (error) {
    console.error("❌ Login error:", error);
    res.status(500).json({ message: "Internal server error during login" });
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

app.post("/clearMessages", async (req, res) => {
  const { username } = req.body;

  if (!username) {
    return res.status(400).json({ error: "Username is required." });
  }

  try {
    await myMessages.deleteMany({ username });
    console.log(`✅ Messages cleared for user: ${username}`);
    res.status(200).json({ message: "Messages cleared successfully." });
  } catch (error) {
    console.error("❌ Failed to clear messages:", error);
    res.status(500).json({ error: "Failed to clear messages." });
  }
});

app.listen(PORT, async () => {
    await connectDb();
    console.log(`Server is running on http://localhost:${PORT}`);
  });
  