from rest_framework.views import APIView,Response
import json
from main.interface import main
import traceback

class MainView(APIView):

    def post(self, request):

        param = request.POST.get('param')
        model = request.POST.get('model')
        data = request.POST.get('data')
        context = dict()


        context['err_code'] = 0
        if data is None or model is None or param is None:
            context['err_code'] = 2002
            context['msg'] = "参数不足"
            return Response(context)
        try:
            if isinstance(param, str):
                param = json.loads(param)
            context = main(data, model, param)

            return Response(context)
        except (Exception, BaseException) as e:
            exstr = traceback.format_exc()
            print(exstr)
            print(e)
            context['err_code'] = 6100
            context['msg'] = "未知错误，请联系管理员"
            return Response(context)

