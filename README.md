# Server with RAG

## Main Info about the server
- The server is built with Flask framework.
- CORS is enabled to accept same-source(localhost) requests as well
- To run the server use python *app.py*
- To send a request via POSTMAN:
    1. Method: POST
    2. URI: http://127.0.0.1:5000/askllm
    3. Headers should include "Content-Type": "application/json"
    4. The body should include raw json object with a required "question" key (example {"question": "should i use flask for this exercise"}). In case this key is not provided error message is sent with 400 code. Sending an empty string will also result in an error

- N.B. other files are irrelevant ! (llama-llm had problems with indexing with FAISS)


## Tests 
1. Test edge cases - an http request sent w/o body VS not in JSON format VS empty string sent (getting 400 error status)

2. Test different inputs: open-ended questions ("what are the main requirements"), questions that require reasoning (one of the requirements was to create a server, on entering "should i use Flask for this exercise" one of the responses was: "Based on the provided context, it might be recommended to use Flask for developing the Python service because Flask is a popular web framework in Python for building RESTful APIs, which aligns with the main requirement of using REST API for receiving questions and returning answers. Using Flask will help in implementing the service efficiently and effectively.") etc.

3. Test extensively for HALLUCINATIONS ! (Question: "What is my favourite color", gives an answer: "The provided context does not contain information about the author's favorite color, so the service would not be able to provide an answer to the question \"What is my favourite color?\"", which is a good one)

