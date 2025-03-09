(function() {
  let canvas2sever = new Array() 
  var datetime = 0;
  var canvas = document.querySelector("#canvas");
  var context = canvas.getContext("2d");
  canvas.width = 560;
  canvas.height = 280;
  var Mouse = { x: 0, y: 0 };
  var lastMouse = { x: 0, y: 0 };
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.color = "#000000";
  context.lineWidth = 2;
  context.lineJoin = context.lineCap = "round";
  context.canvas.style.touchAction = "none";

  var name = sessionStorage.getItem('username')
  console.log(sessionStorage)
functionsCanvas();
//upsetCanvas();

//鼠标画画
canvas.addEventListener( "mousemove",
  function(e) {
    lastMouse.x = Mouse.x;
    lastMouse.y = Mouse.y;
    Mouse.x = e.pageX - this.offsetLeft;
    Mouse.y = e.pageY - this.offsetTop;
}, false);
canvas.addEventListener("mousedown",
  function(e) {
    canvas.addEventListener("mousemove", onDraw, false);
    console.log(Mouse.x, Mouse.y,'1')
    canvas2sever.push([Mouse.x, Mouse.y,'1'])
}, false);
canvas.addEventListener("mouseup",
  function() {
    canvas.removeEventListener("mousemove", onDraw, false);
    console.log(Mouse.x, Mouse.y,'2')
    canvas2sever.push([Mouse.x, Mouse.y,'2'])
}, false);

//笔画画
canvas.addEventListener( "pointermove",
  function(e) {
    if (e.pointerType !== 'pen') {
      return
    }
    lastMouse.x = Mouse.x;
    lastMouse.y = Mouse.y;
 //   console.log(e)
    Mouse.x = e.pageX - this.offsetLeft;
    Mouse.y = e.pageY - this.offsetTop;
}, false);
canvas.addEventListener("pointermove",
  function(e) {
    if (e.pointerType !== 'pen') {
      return
    }
    canvas.addEventListener("pointermove", onDraw, false);
    console.log(Mouse.x, Mouse.y,'0')
    canvas2sever.push([Mouse.x, Mouse.y,'0'])
}, false);
canvas.addEventListener("pointerup",
  function(e) {
    if (e.pointerType !== 'pen') {
      return
    }
    canvas.removeEventListener("pointermove", onDraw, false);
    console.log(Mouse.x, Mouse.y,'2')
    canvas2sever.push([Mouse.x, Mouse.y,'2'])
}, false);



//触摸画画
canvas.addEventListener( "touchmove",
  function(e) {
    lastMouse.x = Mouse.x;
    lastMouse.y = Mouse.y;
    Mouse.x = e.touches[0].pageX - this.offsetLeft;
    Mouse.y = e.touches[0].pageY - this.offsetTop;
}, false);
canvas.addEventListener("touchmove",
  function(e) {
    canvas.addEventListener("touchmove", onDraw, false);
    console.log(Mouse.x, Mouse.y,'0')
    canvas2sever.push([Mouse.x, Mouse.y,'0'])
}, false);
canvas.addEventListener("touchend",
  function() {
    canvas.removeEventListener("touchmove", onDraw, false);
    console.log(Mouse.x, Mouse.y,'2')
    canvas2sever.push([Mouse.x, Mouse.y,'2'])
}, false);




/* Canvas Draw */
var onDraw = function() {
  // context.lineWidth = context.lineWidth;
  painting  = true;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.strokeStyle = context.color;
  context.beginPath();
  context.moveTo(lastMouse.x, lastMouse.y);
  context.lineTo(Mouse.x, Mouse.y);
//  console.log("画了");
  var date = new Date();
  var date1 = date.getTime();
  if (date1-datetime > 50) {
    console.log(Mouse.x, Mouse.y,'0');
    canvas2sever.push([Mouse.x, Mouse.y,'0'])
    datetime = date1;
  }
  context.closePath();
  context.stroke();
};
/* This function clears the box */
//  画板的操作
function functionsCanvas() {

  var clearButton = $("#clearButton");
  clearButton.on("click", function() {
    console.log(canvas2sever);
    canvas2sever = new Array() 
    context.clearRect(0, 0, 560, 560);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    location.reload();
});


  var upsetButton = $("#upsetButton");
  upsetButton.on("click", function() {
  console.log(name);
    $.ajax({
          type: "POST",
          url:  "../predict/",
          data: name +'??lcq??'+canvas2sever.toString()  ,
          success: function(data){
            if(data == 'r'){
              location.reload();
            }
            else if(data == 'g'){
            $('#result').text('是本人');
            }
            else if(data == 'f'){
              $('#result').text('不是本人');
              }
        }})
});

  var upsetButton = $("#deleteButton");
  upsetButton.on("click", function() {
  console.log("deleteButton");
  $.ajax({
        type: "get",
        url:  "../delete/",
        // data: name +'??lcq??'+canvas2sever.toString()  ,
        success: function(data){
          if(data == 'success'){
            location.reload();
          }
      }})
  sessionStorage.removeItem(name)
});
  var upsetButton = $("#ClearAllButton");
  upsetButton.on("click", function() {
  //console.log("ClearAllButton");
  var r=confirm("该操作将导致该id名下所有数据均删除，是否继续？");
  if (r==false){window.history.back(-1); return}
  $.ajax({
        type: "get",
        url:  "../delete_all/",
        success: function(data){
          if(data == 'success'){
            location.reload();
          }
      }})
  sessionStorage.removeItem(name)
});






}})();