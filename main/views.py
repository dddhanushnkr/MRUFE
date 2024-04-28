from django.shortcuts import render
from django.http.response import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from main.detect import getExpression
from django.shortcuts import render
from django.http import HttpResponseServerError
import json

def index(request):
    return render(request, 'index.html')

@csrf_exempt 
def expression(request):
    uri = json.loads(request.body)['image_uri']
    expression = getExpression(uri)
    return JsonResponse({"mood": expression})

