from flask import Flask, request, jsonify
import json
import evaluate

app = Flask(__name__)


@app.route('/hello')
def hello_world():
    return jsonify(text='Hello World!')

@app.route('/process/nlp/request', methods=['POST'] )
def setence_nlp():
    my_json = '[' + json.dumps(request.get_json()) + ']'
    temp_way = ''
    print(my_json.lower().find("buy"))
    if my_json.lower().find("buy") != -1:
        temp_way = "buy"

    elif my_json.lower().find("call") != -1:
        temp_way = "buy"

    elif my_json.lower().find("sell") != -1:
        temp_way = "sell"

    elif my_json.lower().find("put") != -1:
        temp_way = "sell"

    print(temp_way)

    filename = 'test.json'
    with open(filename,'w') as file_obj:
        file_obj.write(my_json)
        #json.dump(my_json, file_obj)
    nlp_result = evaluate.solve_function()
    nlp_result = '{' +nlp_result + '"OPERATMODE": "' + temp_way + '"}'

    return nlp_result

if __name__ == '__main__':
    app.run(host="0.0.0.0") #任何主机都可以访问
