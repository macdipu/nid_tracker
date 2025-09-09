import 'dart:math' as math;
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]).then((_) => runApp(const MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'NID Detector',
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const NidLiveDetectPage(),
    );
  }
}

class NidLiveDetectPage extends StatefulWidget {
  const NidLiveDetectPage({super.key});

  @override
  State<NidLiveDetectPage> createState() => _NidLiveDetectPageState();
}

class _NidLiveDetectPageState extends State<NidLiveDetectPage> {
  final _controller = YOLOViewController();
  List<YOLOResult> _results = const [];
  Map<String, YOLOResult> _latestByLabel = {};
  double _zoom = 1.0;

  // Absolute model file path copied from assets
  String? _modelFilePath;

  // Labels loaded from assets/labels.txt
  List<String> _labels = const [];
  bool _labelsLoaded = false;
  double? _fps;
  int _lastEventMs = 0;

  // static const List<String> _labels = [
  //   'dob',
  //   'father_name',
  //   'image',
  //   'mother_name',
  //   'name',
  //   'name_bn',
  //   'nid_front_image',
  //   'nid_no',
  //   'signature',
  // ];

  @override
  void initState() {
    super.initState();
    _prepareModelPath();
    _loadLabels();
    // Apply more permissive thresholds to ensure early visibility
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _controller.setThresholds(confidenceThreshold: 0.60, iouThreshold: 0.50, numItemsThreshold: 100);
    });
  }

  Future<void> _prepareModelPath() async {
    try {
      final dir = await getApplicationSupportDirectory();
      final modelsDir = Directory('${dir.path}/models');
      if (!await modelsDir.exists()) {
        await modelsDir.create(recursive: true);
      }
      final outFile = File('${modelsDir.path}/model.tflite');
      // Always copy on first run; overwrite if file missing or zero sized
      final data = await rootBundle.load('assets/model.tflite');
      if (!await outFile.exists() || (await outFile.length()) != data.lengthInBytes) {
        await outFile.writeAsBytes(data.buffer.asUint8List(), flush: true);
      }
      if (mounted) setState(() => _modelFilePath = outFile.path);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Model copy failed: $e')),
      );
    }
  }

  Future<void> _loadLabels() async {
    try {
      final txt = await rootBundle.loadString('assets/labels.txt');
      final lines = txt.split(RegExp(r'\r?\n')).where((l) => l.trim().isNotEmpty).toList();
      if (mounted) {
        setState(() {
          _labels = lines;
          _labelsLoaded = true;
          // Recompute latest detections mapped by label using current _results
          _latestByLabel = {
            for (final r in _results)
              if (_labels.contains(_displayName(r))) _displayName(r): r,
          };
        });
      }
    } catch (_) {
      // ignore if labels not present
    }
  }

  String _displayName(YOLOResult r) {
    final name = r.className.trim();
    final looksNumeric = RegExp(r'^\d+$').hasMatch(name);
    if (_labelsLoaded && (name.isEmpty || looksNumeric) && r.classIndex >= 0 && r.classIndex < _labels.length) {
      return _labels[r.classIndex];
    }
    return name.isEmpty && r.classIndex >= 0 && r.classIndex < _labels.length
        ? _labels[r.classIndex]
        : (name.isEmpty ? 'class_${r.classIndex}' : name);
  }

  @override
  Widget build(BuildContext context) {
    final ready = _modelFilePath != null; // wait until model file is ready
    return Scaffold(
      appBar: AppBar(
        title: const Text('NID Front - Live Detection'),
        actions: [
          IconButton(
            tooltip: 'Switch camera',
            icon: const Icon(Icons.cameraswitch),
            onPressed: () => _controller.switchCamera(),
          ),
        ],
      ),
      body: !ready
          ? const Center(child: Text('Preparing model…'))
          : LayoutBuilder(
        builder: (context, constraints) {
          final screenSize = Size(constraints.maxWidth, constraints.maxHeight);
          return Stack(
            fit: StackFit.expand,
            children: [
              YOLOView(
                modelPath: _modelFilePath!,
                task: YOLOTask.detect,
                controller: _controller,
                // Enable native overlay too, for quicker visual feedback
                showNativeUI: false,
                useGpu: true,
                confidenceThreshold: 0.60,
                iouThreshold: 0.50,
                streamingConfig: const YOLOStreamingConfig.minimal(),
                onResult: (List<YOLOResult> results) {
                  final now = DateTime.now().millisecondsSinceEpoch;
                  // Make a fresh copy to force painter repaint comparisons
                  final listCopy = List<YOLOResult>.from(results);
                  setState(() {
                    _lastEventMs = now;
                    _results = listCopy;
                    _latestByLabel = {
                      for (final r in listCopy)
                        if (_labels.contains(_displayName(r))) _displayName(r): r,
                    };
                  });
                },
                onStreamingData: (Map<String, dynamic> stream) {
                  // Fallback path: parse raw stream when onResult is bypassed
                  try {
                    final now = DateTime.now().millisecondsSinceEpoch;
                    final dets = (stream['detections'] as List?) ?? const [];
                    final parsed = dets.whereType<Map>().map((m) => YOLOResult.fromMap(m)).toList();
                    setState(() {
                      _lastEventMs = now;
                      _fps = (stream['fps'] is num) ? (stream['fps'] as num).toDouble() : _fps;
                      _results = parsed; // already a fresh list
                      _latestByLabel = {
                        for (final r in parsed)
                          if (_labels.contains(_displayName(r))) _displayName(r): r,
                      };
                    });
                  } catch (_) {/*ignore parse errors*/}
                },
                onPerformanceMetrics: (m) {
                  setState(() { _fps = m.fps; _lastEventMs = DateTime.now().millisecondsSinceEpoch; });
                },
                onZoomChanged: (z) => setState(() => _zoom = z),
              ),
              // Our overlay for custom styling/colors and summary chips
              CustomPaint(
                painter: _ResultsPainter(results: _results, screenSize: screenSize, nameFor: _displayName),
              ),
              // Status banner (FPS / No detections yet)
              Positioned(
                top: 12,
                left: 12,
                right: 12,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _StatusBanner(fps: _fps, lastEventMs: _lastEventMs),
                    const SizedBox(height: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.45),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text('Detections: ${_results.length}', style: const TextStyle(color: Colors.white)),
                    ),
                  ],
                ),
              ),
              Align(
                alignment: Alignment.bottomCenter,
                child: _buildBottomPanel(),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildBottomPanel() {
    return SafeArea(
      top: false,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.45),
          borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
        ),
        child: DefaultTextStyle(
          style: const TextStyle(color: Colors.white, fontSize: 14),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Text('Detected fields', style: TextStyle(fontWeight: FontWeight.bold)),
                  const Spacer(),
                  Text('Zoom ${_zoom.toStringAsFixed(1)}x'),
                ],
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  for (final key in _labels)
                    _FieldChip(
                      label: key,
                      value: _latestByLabel[key]?.confidence != null
                          ? '${(_latestByLabel[key]!.confidence * 100).toStringAsFixed(0)}%'
                          : '—',
                      color: _colorForLabel(key),
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Color _colorForLabel(String label) {
    if (_labels.isEmpty) return Colors.grey;
    final idx = _labels.indexOf(label);
    final hue = (idx / _labels.length) * 360.0;
    return HSLColor.fromAHSL(1.0, hue, 0.8, 0.5).toColor();
  }
}

class _ResultsPainter extends CustomPainter {
  final List<YOLOResult> results;
  final Size screenSize;
  final String Function(YOLOResult) nameFor;

  _ResultsPainter({required this.results, required this.screenSize, required this.nameFor});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    final textPainter = TextPainter(textDirection: TextDirection.ltr);

    for (final r in results) {
      final className = nameFor(r);
      final color = _colorFromLabel(className);
      paint.color = color;

      final nb = r.normalizedBox;
      final rect = Rect.fromLTWH(
        nb.left * size.width,
        nb.top * size.height,
        nb.width * size.width,
        nb.height * size.height,
      );

      canvas.drawRect(rect, paint);

      final label = '$className ${(r.confidence * 100).toStringAsFixed(0)}%';
      textPainter.text = TextSpan(
        text: label,
        style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600),
      );
      textPainter.layout();
      final tp = Offset(rect.left, math.max(0, rect.top - textPainter.height - 2));
      final bgRect = Rect.fromLTWH(tp.dx - 2, tp.dy - 2, textPainter.width + 4, textPainter.height + 4);
      final bgPaint = Paint()..color = Colors.black.withValues(alpha: 0.55);
      canvas.drawRRect(RRect.fromRectAndRadius(bgRect, const Radius.circular(4)), bgPaint);
      textPainter.paint(canvas, tp);
    }
  }

  @override
  bool shouldRepaint(covariant _ResultsPainter oldDelegate) {
    return oldDelegate.results != results || oldDelegate.screenSize != screenSize || oldDelegate.nameFor != nameFor;
  }

  Color _colorFromLabel(String label) {
    const labels = [
      'dob',
      'father_name',
      'image',
      'mother_name',
      'name',
      'name_bn',
      'nid_front_image',
      'nid_no',
      'signature',
    ];
    final idx = math.max(0, labels.indexOf(label));
    final hue = (idx / labels.length) * 360.0;
    return HSLColor.fromAHSL(1.0, hue, 0.8, 0.5).toColor();
  }
}

class _FieldChip extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _FieldChip({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color, width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(color: Colors.white)),
          const SizedBox(width: 8),
          Text(value, style: const TextStyle(color: Colors.white70)),
        ],
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  final double? fps;
  final int lastEventMs;
  const _StatusBanner({required this.fps, required this.lastEventMs});

  @override
  Widget build(BuildContext context) {
    final now = DateTime.now().millisecondsSinceEpoch;
    final stale = now - lastEventMs > 3000; // No events in >3s
    final text = fps != null && !stale
        ? 'FPS ${fps!.toStringAsFixed(1)}'
        : 'No detections/events yet. Check camera permission and model path/format.';
    return Align(
      alignment: Alignment.topLeft,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.45),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(text, style: const TextStyle(color: Colors.white)),
      ),
    );
  }
}
