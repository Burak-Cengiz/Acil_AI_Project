from django.shortcuts import render
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline,AutoModelForMaskedLM



model_name = 'deprem-ml/multilabel_earthquake_tweet_intent_bert_base_turkish_cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

pipe = pipeline("fill-mask", model="NurDrmz/AcilAI_Model")
tokenizer2 = AutoTokenizer.from_pretrained("NurDrmz/AcilAI_Model")
model2 = AutoModelForMaskedLM.from_pretrained("NurDrmz/AcilAI_Model")


def process_input_text(text_input, tokenizer):
    inputs = tokenizer(text_input, return_tensors="pt")
    return inputs

def make_prediction(text_input, tokenizer, model):
    inputs = process_input_text(text_input, tokenizer)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class



def predictor(request):
    if request.method == 'POST':
        print("Predictor1 function called.")

        text_input = request.POST.get('text_input', '')
        
        # Giriş metni için işleme fonksiyonunu kullanarak hazırlık yap
        inputs = process_input_text(text_input, tokenizer)

        # Modeli kullanarak tahmin işlemlerini yap
        outputs = model(**inputs)
        predicted_scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()

        # Modelin kategorilerini tanımlayın
        class_labels = ['Alakasiz', 'Barinma', 'Elektronik', 'Giysi', 'Kurtarma', 'Lojistik', 'Saglik', 'Su', 'Yagma', 'Yemek']

        # Skorları kategorilere atayarak bir sözlük oluşturun
        prediction_dict = dict(zip(class_labels, predicted_scores))

        # Ağırlıkları büyükten küçüğe sıralayın
        sorted_predictions = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)

        # Sıralı tahmin sonucunu gönder
        context = {'sorted_predictions': sorted_predictions, 'text_input': text_input}
        print(context)
        return render(request, 'main.html', context)

    return render(request, 'main.html')



# #/////////////////////////////////

def process_input_text2(text_input, tokenizer2):
    inputs = tokenizer2(text_input, return_tensors="pt")
    return inputs

def make_prediction2(text_input, tokenizer2, model2):
    inputs = process_input_text2(text_input, tokenizer2)
    outputs = model2(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

def predictor2(request):
    if request.method == 'POST':
        print("Predictor2 function called.")

        text_input = request.POST.get('text_input', '')
        
        # Giriş metni için işleme fonksiyonunu kullanarak hazırlık yap
        inputs = process_input_text2(text_input, tokenizer2)

        # Modeli kullanarak tahmin işlemlerini yap
        outputs = model2(**inputs)
        predicted_scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
        print(predicted_scores)
        # Modelin kategorilerini tanımlayın
        class_labels = ['Alakasiz', 'Barinma', 'Elektronik', 'Giysi', 'Kurtarma', 'Lojistik', 'Saglik', 'Su', 'Yagma', 'Yemek']

        # Skorları kategorilere atayarak bir sözlük oluşturun
        prediction_dict = dict(zip(class_labels, predicted_scores))
        print(prediction_dict)
        # Ağırlıkları büyükten küçüğe sıralayın
        sorted_predictions = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)

        # Sıralı tahmin sonucunu gönder
        context = {'sorted_predictions2': sorted_predictions}
        print(context)

        return render(request, 'main.html', context)

    return render(request, 'main.html')


# from django.shortcuts import render
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForMaskedLM

# def get_model1_prediction(text_input, tokenizer, model):
#     inputs = tokenizer(text_input, return_tensors="pt")
#     outputs = model(**inputs)
#     predicted_scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
#     return predicted_scores

# def get_model2_prediction(text_input, tokenizer, model):
#     inputs = tokenizer(text_input, return_tensors="pt")
#     outputs = model(**inputs)
#     predicted_scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
#     return predicted_scores

# def combined_predictor(request):
#     if request.method == 'POST':
#         text_input = request.POST.get('text_input', '')

#         # Load or initialize your models and tokenizers
#         tokenizer1 = AutoTokenizer.from_pretrained("deprem-ml/multilabel_earthquake_tweet_intent_bert_base_turkish_cased")
#         model1 = AutoModelForSequenceClassification.from_pretrained("deprem-ml/multilabel_earthquake_tweet_intent_bert_base_turkish_cased")

#         tokenizer2 = AutoTokenizer.from_pretrained("NurDrmz/AcilAI_Model")
#         model2 = AutoModelForMaskedLM.from_pretrained("NurDrmz/AcilAI_Model")

#         # Get predictions from both models
#         predictions_model1 = get_model1_prediction(text_input, tokenizer1, model1)
#         predictions_model2 = get_model2_prediction(text_input, tokenizer2, model2)

#         # Combine or process the predictions as needed
#         print(predictions_model1)
        
#         # Pass the results to the template
#         context = {
#             'text_input': text_input,
#             'predictions_model1': predictions_model1,
#             'predictions_model2': predictions_model2,
#         }

#         return render(request, 'main.html', context)

#     return render(request, 'main.html')
