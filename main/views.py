from rest_framework.views import APIView,Response
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
            return context


        return Response(context)

