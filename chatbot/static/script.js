document.getElementById("send-button").addEventListener("click", sendMessage);

document
  .getElementById("user-input")
  .addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });

function sendMessage() {
  const userInput = document.getElementById("user-input").value;
  if (!userInput.trim()) return;

  // Display the user's message
  addMessage(userInput, "user-message");

  // Send the message to the server
  fetch("http://18.234.161.171/chatbot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: userInput,
      metadata: {
        organization_name: "Infojini",
      },
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.response) {
        addMessage(data.response, "bot-message");
      } else {
        addMessage(data.message, "bot-message");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      addMessage("An error occurred. Please try again.", "bot-message");
    });

  // Clear the input field
  document.getElementById("user-input").value = "";
}

function addMessage(content, className) {
  const chatWindow = document.getElementById("chat-window");
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", className);

  // Format URLs in the content
  messageDiv.innerHTML = formatMessageWithLinks(content);

  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function formatMessageWithLinks(message) {
  const urlRegex =
    /(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?/g;
  return message.replace(urlRegex, (url) => {
    // Ensure the URL is clickable by adding "http://" if missing
    let clickableUrl = url;
    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      clickableUrl = `http://${url}`;
    }
    return `<a href="${clickableUrl}" target="_blank">${url}</a>`;
  });
}
