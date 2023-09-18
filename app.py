import gradio as gr
import torch

from diffusers import AutoPipelineForInpainting, UNet2DConditionModel
import diffusers
from share_btn import community_icon_html, loading_icon_html, share_js

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.scheduler = scheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", fp16=True, **add_kwargs)

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

def predict(dict, prompt="", negative_prompt="", guidance_scale=7.5, steps=20, strength=1.0, scheduler="EulerDiscreteScheduler"):
    if negative_prompt == "":
        negative_prompt = None
    scheduler_class_name = scheduler.split("-")[0]

    add_kwargs = {}
    if len(scheduler.split("-")) > 1:
        add_kwargs["use_karras"] = True
    if len(scheduler.split("-")) > 2:
        add_kwargs["algorithm_type"] = "sde-dpmsolver++"

    scheduler = getattr(diffusers, scheduler_class_name)
    pipe.scheduler = scheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", **add_kwargs)
    
    init_image = dict["image"].convert("RGB").resize((1024, 1024))
    mask = dict["mask"].convert("RGB").resize((1024, 1024))
    
    output = pipe(prompt = prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask, guidance_scale=guidance_scale, num_inference_steps=int(steps), strength=strength)
    
    return output.images[0], gr.update(visible=True)


css = '''
.gradio-container{max-width: 1100px !important}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; max-width: 13rem; margin-left: auto;}
div#share-btn-container > div {flex-direction: row;background: black;align-items: center}
#share-btn-container:hover {background-color: #060606}
#share-btn {all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.5rem !important; padding-bottom: 0.5rem !important;right:0;}
#share-btn * {all: unset}
#share-btn-container div:nth-child(-n+2){width: auto !important;min-height: 0px !important;}
#share-btn-container .wrap {display: none !important}
#share-btn-container.hidden {display: none!important}
#prompt input{width: calc(100% - 160px);border-top-right-radius: 0px;border-bottom-right-radius: 0px;}
#run_button{position:absolute;margin-top: 11px;right: 0;margin-right: 0.8em;border-bottom-left-radius: 0px;
    border-top-left-radius: 0px;}
#prompt-container{margin-top:-18px;}
#prompt-container .form{border-top-left-radius: 0;border-top-right-radius: 0}
#image_upload{border-bottom-left-radius: 0px;border-bottom-right-radius: 0px}
'''

image_blocks = gr.Blocks(css=css, elem_id="total-container")
with image_blocks as demo:
    gr.HTML(read_content("header.html"))
    with gr.Row():
                with gr.Column():
                    image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload",height=400)
                    with gr.Row(elem_id="prompt-container", mobile_collapse=False, equal_height=True):
                        with gr.Row():
                            prompt = gr.Textbox(placeholder="Your prompt (Что вы хотите, чтобы ИИ генерировал в области маски?)", show_label=False, elem_id="prompt")
                            btn = gr.Button("Перерисовать!", elem_id="run_button")
                    
                    with gr.Accordion(label="Расширенные настройки", open=False):
                        with gr.Row(mobile_collapse=False, equal_height=True):
                            guidance_scale = gr.Number(value=7.5, minimum=1.0, maximum=20.0, step=0.1, label="guidance_scale")
                            steps = gr.Number(value=20, minimum=10, maximum=30, step=1, label="steps")
                            strength = gr.Number(value=0.99, minimum=0.01, maximum=0.99, step=0.01, label="Сила")
                            negative_prompt = gr.Textbox(label="Что вы не хотите, чтобы ИИ генерировал в области маски?", placeholder="Your negative prompt", info="what you don't want to see in the image")
                        with gr.Row(mobile_collapse=False, equal_height=True):
                            schedulers = ["DEISMultistepScheduler", "HeunDiscreteScheduler", "EulerDiscreteScheduler", "DPMSolverMultistepScheduler", "DPMSolverMultistepScheduler-Karras", "DPMSolverMultistepScheduler-Karras-SDE"]
                            scheduler = gr.Dropdown(label="Schedulers", choices=schedulers, value="EulerDiscreteScheduler")
                        
                with gr.Column():
                    image_out = gr.Image(label="Output", elem_id="output-img", height=400)
                    with gr.Group(elem_id="share-btn-container", visible=False) as share_btn_container:
                        community_icon = gr.HTML(community_icon_html)
                        loading_icon = gr.HTML(loading_icon_html)
                        share_button = gr.Button("Share to community", elem_id="share-btn",visible=True)
            

    btn.click(fn=predict, inputs=[image, prompt, negative_prompt, guidance_scale, steps, strength, scheduler], outputs=[image_out, share_btn_container], api_name='run')
    prompt.submit(fn=predict, inputs=[image, prompt, negative_prompt, guidance_scale, steps, strength, scheduler], outputs=[image_out, share_btn_container])
    share_button.click(None, [], [], _js=share_js)

    gr.Examples(
                examples=[
                    ["./imgs/aaa (8).png"],
                    ["./imgs/download (1).jpeg"],
                    ["./imgs/0_oE0mLhfhtS_3Nfm2.png"],
                    ["./imgs/02_HubertyBlog-1-1024x1024.jpg"],
                    ["./imgs/jdn_jacques_de_nuce-1024x1024.jpg"],
                    ["./imgs/c4ca473acde04280d44128ad8ee09e8a.jpg"],
                    ["./imgs/canam-electric-motorcycles-scaled.jpg"],
                    ["./imgs/e8717ce80b394d1b9a610d04a1decd3a.jpeg"],
                    ["./imgs/Nature___Mountains_Big_Mountain_018453_31.jpg"],
                    ["./imgs/Multible-sharing-room_ccexpress-2-1024x1024.jpeg"],
                ],
                fn=predict,
                inputs=[image],
                cache_examples=False,
    )
    gr.HTML(
        """
            <div class="footer">
              
                <p style='text-align: center'>Будь в курсе обновлений <a href='https://vk.com/public221489796'>ПОДПИСАТЬСЯ</a></p>
            </div>
        """
    )

image_blocks.queue(max_size=25).launch(debug=True, max_threads=True, share=True, inbrowser=True)
