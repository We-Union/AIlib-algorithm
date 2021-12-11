from rest_framework.views import APIView,Response
import json
from main.interface import main
class MainView(APIView):

    def post(self, request):

        param = request.POST.get('param')
        model = request.POST.get('model')
        data = request.POST.get('data')
        context = dict()
        if isinstance(param, str):
            param = json.loads(param)

        context['err_code'] = 0
        print("param:",param)
        print("model",model)
        print("data",data)
        if data is None or model is None or param is None:
            context['err_code'] = 2002
            context['msg'] = "参数不足"
            return Response(context)
        
        context = main(data, model, param)

        return Response(context)

